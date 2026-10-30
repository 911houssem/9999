"""
Arabic voice agent with web UI — single file.
Browser mic -> WebSocket -> Silero VAD -> Jais-ASR -> Jais 3 (vLLM) -> Audar-TTS -> WebSocket -> browser speaker

Setup:
  # 1) LLM server (separate terminal, GPU)
  pip install vllm
  vllm serve inception42/Jais-3-7B --trust-remote-code --dtype bfloat16 \
      --reasoning-parser k2_horizon --port 8000

  # 2) This app
  pip install "pipecat-ai[silero,websocket,openai]" fastapi uvicorn transformers accelerate torch \
      soundfile soxr numpy llama-cpp-python neucodec huggingface_hub
  wget https://huggingface.co/audarai/Audar-TTS-V1-Turbo/resolve/main/samples/demo_male_1_source.wav -O ref.wav
  python voice_agent_ui.py          # then open http://localhost:7860

Env: REF_WAV, REF_TEXT, LLM_MODEL, LLM_BASE_URL, ASR_MODEL, TTS_GGUF, TTS_GPU_LAYERS, SYSTEM_PROMPT, PORT
"""
import asyncio
import io
import json
import os
import re
import threading

import numpy as np
import soundfile as sf
import soxr
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

try:  # name differs between pipecat versions
    from pipecat.frames.frames import InterruptionFrame
except ImportError:
    from pipecat.frames.frames import StartInterruptionFrame as InterruptionFrame

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.utils.time import time_now_iso8601

# ----------------------------- config ---------------------------------------
ASR_MODEL = os.getenv("ASR_MODEL", "IFM/Jais-ASR")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "inception42/Jais-3-7B")
TTS_REPO = os.getenv("TTS_REPO", "audarai/Audar-TTS-V1-Turbo")
TTS_GGUF = os.getenv("TTS_GGUF", "Audar-TTS-V1-Turbo-Q4_K_M.gguf")
TTS_GPU_LAYERS = int(os.getenv("TTS_GPU_LAYERS", "0"))
REF_WAV = os.getenv("REF_WAV", "ref.wav")
REF_TEXT = os.getenv("REF_TEXT", "")
PORT = int(os.getenv("PORT", "7860"))
IN_SR, TTS_SR = 16000, 24000

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "أنت مساعد صوتي ذكي ودود. تكلّم بالعربية البسيطة وبنفس لهجة المستخدم. "
    "اجعل ردودك قصيرة جداً (جملة أو جملتين) لأنها ستُنطق صوتياً. "
    "لا تستخدم قوائم أو رموز أو تنسيق Markdown أو إيموجي.",
)


# ----------------------------- engines (loaded once) -------------------------
class ASREngine:
    def __init__(self):
        from transformers import pipeline

        logger.info("Loading Jais-ASR ...")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        self.lock = threading.Lock()

    def transcribe(self, wav16k: np.ndarray) -> str:
        with self.lock:
            out = self.pipe({"raw": wav16k, "sampling_rate": IN_SR}, max_new_tokens=256)
        return (out["text"] if isinstance(out, dict) else str(out)).strip()


class TTSEngine:
    def __init__(self, asr: ASREngine):
        from llama_cpp import Llama
        from neucodec import NeuCodec

        logger.info("Loading Audar-TTS ...")
        gguf = hf_hub_download(TTS_REPO, TTS_GGUF)
        self.llm = Llama(model_path=gguf, n_ctx=4096, n_gpu_layers=TTS_GPU_LAYERS, verbose=False)
        self.codec = NeuCodec.from_pretrained("neuphonic/neucodec").eval()
        self.lock = threading.Lock()

        ref_text = REF_TEXT
        if not ref_text:
            wav, sr = sf.read(REF_WAV, dtype="float32")
            wav = wav.mean(axis=1) if wav.ndim > 1 else wav
            wav = soxr.resample(wav, sr, IN_SR) if sr != IN_SR else wav
            ref_text = asr.transcribe(wav)
            logger.info(f"Reference transcript: {ref_text}")
        codes = self.codec.encode_code(REF_WAV).squeeze().tolist()
        ref_speech = "".join(f"<|speech_{c}|>" for c in codes)
        self.prefix = (
            "user: Convert the text to speech:"
            f"<|REF_TEXT_START|>{ref_text}<|REF_TEXT_END|>"
            f"<|REF_SPEECH_START|>{ref_speech}<|REF_SPEECH_END|>"
        )
        self.end_tok = self.llm.tokenize(b"<|TARGET_CODES_END|>", add_bos=False, special=True)[0]

    def synth(self, text: str) -> bytes:
        prompt = (
            f"{self.prefix}<|TARGET_TEXT_START|>{text}<|TARGET_TEXT_END|>"
            "\nassistant:<|TARGET_CODES_START|>"
        )
        with self.lock:
            self.llm.reset()
            toks = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
            ids = []
            for tid in self.llm.generate(toks, temp=0.7, top_k=40, top_p=0.9, repeat_penalty=1.1):
                if tid == self.end_tok or len(ids) >= 2048:
                    break
                ids.append(tid)
            out = "".join(self.llm.detokenize([t], special=True).decode("utf-8", "ignore") for t in ids)
            codes = [int(x) for x in re.findall(r"<\|speech_(\d+)\|>", out)]
            if not codes:
                return b""
            wav = self.codec.decode_code(torch.tensor(codes)[None, None, :]).cpu().numpy()[0, 0, :]
        return (np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes()


# ----------------------------- pipecat services ------------------------------
class JaisASRService(SegmentedSTTService):
    def __init__(self, engine: ASREngine, **kwargs):
        super().__init__(**kwargs)
        self._engine = engine

    def can_generate_metrics(self) -> bool:
        return False

    def _to_16k(self, audio: bytes) -> np.ndarray:
        try:
            data, sr = sf.read(io.BytesIO(audio), dtype="float32")
        except Exception:
            data = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            sr = self.sample_rate or IN_SR
        if data.ndim > 1:
            data = data.mean(axis=1)
        return soxr.resample(data, sr, IN_SR) if sr != IN_SR else data

    async def run_stt(self, audio: bytes, *args, **kwargs):
        wav = self._to_16k(audio)
        if len(wav) < IN_SR * 0.3:
            return
        text = await asyncio.to_thread(self._engine.transcribe, wav)
        if text:
            logger.info(f"👤 {text}")
            yield TranscriptionFrame(text, "user", time_now_iso8601())


class AudarTTSService(TTSService):
    def __init__(self, engine: TTSEngine, **kwargs):
        super().__init__(sample_rate=TTS_SR, **kwargs)
        self._engine = engine

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, *args, **kwargs):
        text = re.sub(r"[*_#`>]+", "", text).strip()
        if not text:
            return
        logger.info(f"🤖 {text}")
        yield TTSStartedFrame()
        pcm = await asyncio.to_thread(self._engine.synth, text)
        step = TTS_SR * 2 // 5
        for i in range(0, len(pcm), step):
            yield TTSAudioRawFrame(audio=pcm[i : i + step], sample_rate=TTS_SR, num_channels=1)
        yield TTSStoppedFrame()


# ----------------------------- websocket glue --------------------------------
class PCMSerializer(FrameSerializer):
    """Browser <-> server: binary = int16 PCM mono (16k in / 24k out). Text events go via UIBridge."""

    async def serialize(self, frame: Frame):
        if isinstance(frame, TTSAudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data):
        if isinstance(data, (bytes, bytearray)) and data:
            return InputAudioRawFrame(audio=bytes(data), sample_rate=IN_SR, num_channels=1)
        return None


class UIBridge(FrameProcessor):
    """Forwards transcripts / state to the browser as JSON; passes every frame through untouched."""

    def __init__(self, ws: WebSocket, **kwargs):
        super().__init__(**kwargs)
        self._ws = ws

    async def _send(self, **msg):
        try:
            await self._ws.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            await self._send(type="user", text=frame.text)
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._send(type="user_start")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._send(type="user_stop")
        elif isinstance(frame, InterruptionFrame):
            await self._send(type="interrupt")
        elif isinstance(frame, TTSTextFrame):
            await self._send(type="bot", text=frame.text)
        await self.push_frame(frame, direction)


# ----------------------------- app -------------------------------------------
app = FastAPI()
ENGINES: dict = {}


@app.on_event("startup")
async def load_models():
    asr = await asyncio.to_thread(ASREngine)
    tts = await asyncio.to_thread(TTSEngine, asr)
    ENGINES.update(asr=asr, tts=tts)
    logger.info(f"✅ جاهز — افتح http://localhost:{PORT}")


@app.get("/")
async def index():
    return HTMLResponse(HTML)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    if not ENGINES:
        await ws.send_text(json.dumps({"type": "error", "text": "النماذج ما زالت تُحمَّل"}))
        await ws.close()
        return

    transport = FastAPIWebsocketTransport(
        websocket=ws,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_out_sample_rate=TTS_SR,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=PCMSerializer(),
        ),
    )
    stt = JaisASRService(ENGINES["asr"])
    tts = AudarTTSService(ENGINES["tts"])
    llm = OpenAILLMService(api_key="EMPTY", base_url=LLM_BASE_URL, model=LLM_MODEL)
    context = OpenAILLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    agg = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            UIBridge(ws),
            agg.user(),
            llm,
            tts,
            UIBridge(ws),
            transport.output(),
            agg.assistant(),
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            audio_in_sample_rate=IN_SR,
            audio_out_sample_rate=TTS_SR,
        ),
    )

    @transport.event_handler("on_client_disconnected")
    async def _bye(transport, client):
        await task.cancel()

    try:
        await PipelineRunner(handle_sigint=False).run(task)
    except WebSocketDisconnect:
        pass


# ----------------------------- UI --------------------------------------------
HTML = r"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>المساعد الصوتي</title>
<link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#07070a;--panel:#0f0f14;--line:#23222b;--gold:#d4af6a;--gold2:#8a6d34;--txt:#ece7dc;--mut:#8b8678;--you:#17161d;--bot:#1b170d}
*{box-sizing:border-box;margin:0}
html,body{height:100%}
body{background:radial-gradient(1200px 600px at 50% -10%,#1a1508 0%,var(--bg) 60%);color:var(--txt);font-family:Tajawal,system-ui,sans-serif;display:flex;flex-direction:column;align-items:center;padding:28px 16px;gap:18px}
header{text-align:center}
h1{font-weight:300;font-size:26px;letter-spacing:.5px;color:var(--gold)}
header p{color:var(--mut);font-size:13px;margin-top:4px}
.stage{position:relative;width:240px;height:240px;display:grid;place-items:center;flex:none}
.ring{position:absolute;inset:0;border-radius:50%;border:1px solid var(--gold2);opacity:.35}
.ring.r2{inset:-18px;opacity:.15}
.orb{width:150px;height:150px;border-radius:50%;background:radial-gradient(circle at 35% 30%,#f3dca6,var(--gold) 40%,var(--gold2) 100%);box-shadow:0 0 60px #d4af6a55,inset 0 0 30px #00000066;transition:transform .08s linear,filter .3s}
.stage.idle .orb{filter:grayscale(.85) brightness(.55);animation:breath 4s ease-in-out infinite}
.stage.thinking .orb{animation:spin 1.2s linear infinite;filter:hue-rotate(-12deg)}
@keyframes breath{50%{transform:scale(1.04)}}
@keyframes spin{to{transform:rotate(360deg) scale(1.03)}}
#status{color:var(--gold);font-size:15px;min-height:22px;letter-spacing:.3px}
.controls{display:flex;gap:10px}
button{font-family:inherit;font-size:15px;padding:11px 28px;border-radius:999px;border:1px solid var(--gold2);background:transparent;color:var(--gold);cursor:pointer;transition:.2s}
button:hover{background:#d4af6a18}
button.primary{background:linear-gradient(135deg,var(--gold),var(--gold2));color:#14100a;border:none;font-weight:700}
button:disabled{opacity:.35;cursor:default}
#log{width:min(680px,100%);flex:1;min-height:0;overflow-y:auto;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:14px;display:flex;flex-direction:column;gap:10px}
.msg{max-width:82%;padding:10px 14px;border-radius:14px;line-height:1.7;font-size:15px}
.msg.you{align-self:flex-start;background:var(--you);border:1px solid var(--line)}
.msg.bot{align-self:flex-end;background:var(--bot);border:1px solid #3a2f14;color:#f1e6c8}
.empty{color:var(--mut);text-align:center;margin:auto;font-size:14px}
</style>
</head>
<body>
<header><h1>المساعد الصوتي</h1><p>Jais ASR · Jais 3 · Audar TTS · Pipecat</p></header>
<div class="stage idle" id="stage"><div class="ring r2"></div><div class="ring"></div><div class="orb" id="orb"></div></div>
<div id="status">اضغط «ابدأ» ثم تكلّم</div>
<div class="controls">
  <button class="primary" id="start">ابدأ</button>
  <button id="mute" disabled>كتم الميكروفون</button>
  <button id="stop" disabled>إنهاء</button>
</div>
<div id="log"><div class="empty" id="empty">ستظهر المحادثة هنا</div></div>

<script>
const $=id=>document.getElementById(id);
const stage=$('stage'),orb=$('orb'),statusEl=$('status'),log=$('log');
let ws,micCtx,playCtx,analyser,stream,worklet,muted=false;
let nextT=0,sources=[],level=0,userTalking=false,thinking=false,newBot=true,curBot=null;

const workletSrc=`
class P extends AudioWorkletProcessor{
  constructor(){super();this.b=new Float32Array(512);this.n=0}
  process(i){const x=i[0][0];if(!x)return true;
    for(let k=0;k<x.length;k++){this.b[this.n++]=x[k];
      if(this.n===512){const o=new Int16Array(512);let s=0;
        for(let j=0;j<512;j++){const v=Math.max(-1,Math.min(1,this.b[j]));o[j]=v*32767;s+=v*v}
        this.port.postMessage({pcm:o.buffer,rms:Math.sqrt(s/512)},[o.buffer]);this.n=0}}
    return true}}
registerProcessor('p',P);`;

function setState(){
  const speaking=sources.length>0;
  let cls='idle',txt='جاهز — تكلّم';
  if(!ws){txt='اضغط «ابدأ» ثم تكلّم'}
  else if(speaking){cls='speaking';txt='يتكلّم…'}
  else if(userTalking){cls='listening';txt='أسمعك…'}
  else if(thinking){cls='thinking';txt='يفكّر…'}
  stage.className='stage '+cls;statusEl.textContent=txt;
}
function addMsg(who,text){
  $('empty')&&$('empty').remove();
  const d=document.createElement('div');d.className='msg '+who;d.textContent=text;
  log.appendChild(d);log.scrollTop=log.scrollHeight;return d;
}
function stopPlayback(){sources.forEach(s=>{try{s.onended=null;s.stop()}catch(e){}});sources=[];nextT=0;setState()}

function play(buf){
  const i16=new Int16Array(buf),f=new Float32Array(i16.length);
  for(let i=0;i<i16.length;i++)f[i]=i16[i]/32768;
  const ab=playCtx.createBuffer(1,f.length,24000);ab.copyToChannel(f,0);
  const s=playCtx.createBufferSource();s.buffer=ab;s.connect(analyser);
  nextT=Math.max(nextT,playCtx.currentTime+0.05);s.start(nextT);nextT+=ab.duration;
  sources.push(s);thinking=false;
  s.onended=()=>{sources=sources.filter(x=>x!==s);setState()};setState();
}

async function start(){
  $('start').disabled=true;
  try{
    stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,channelCount:1}});
  }catch(e){statusEl.textContent='لا يوجد إذن للميكروفون';$('start').disabled=false;return}
  playCtx=new AudioContext();analyser=playCtx.createAnalyser();analyser.fftSize=256;analyser.connect(playCtx.destination);
  micCtx=new AudioContext({sampleRate:16000});
  const url=URL.createObjectURL(new Blob([workletSrc],{type:'application/javascript'}));
  await micCtx.audioWorklet.addModule(url);
  worklet=new AudioWorkletNode(micCtx,'p');
  micCtx.createMediaStreamSource(stream).connect(worklet);

  ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
  ws.binaryType='arraybuffer';
  ws.onopen=()=>{$('mute').disabled=false;$('stop').disabled=false;setState()};
  ws.onclose=stop;
  ws.onmessage=e=>{
    if(typeof e.data!=='string'){play(e.data);return}
    const m=JSON.parse(e.data);
    if(m.type==='user'){addMsg('you',m.text);newBot=true;thinking=true;userTalking=false}
    else if(m.type==='user_start'){userTalking=true;stopPlayback()}
    else if(m.type==='user_stop'){userTalking=false;thinking=true}
    else if(m.type==='bot'){
      if(newBot||!curBot){curBot=addMsg('bot','');newBot=false}
      curBot.textContent+=(curBot.textContent?' ':'')+m.text;log.scrollTop=log.scrollHeight}
    else if(m.type==='interrupt'){stopPlayback();newBot=true}
    else if(m.type==='error'){statusEl.textContent=m.text}
    setState();
  };
  worklet.port.onmessage=e=>{
    if(!ws||ws.readyState!==1)return;
    level=Math.max(level*0.8,e.data.rms*6);
    if(!muted)ws.send(e.data.pcm);
  };
  requestAnimationFrame(loop);
}
function stop(){
  try{ws&&ws.close()}catch(e){}
  ws=null;stopPlayback();
  stream&&stream.getTracks().forEach(t=>t.stop());
  micCtx&&micCtx.close();playCtx&&playCtx.close();
  micCtx=playCtx=null;userTalking=thinking=false;muted=false;
  $('start').disabled=false;$('mute').disabled=true;$('stop').disabled=true;$('mute').textContent='كتم الميكروفون';
  setState();
}
function loop(){
  if(!ws)return orb.style.transform='';
  let v=0;
  if(sources.length&&analyser){const d=new Uint8Array(analyser.frequencyBinCount);analyser.getByteFrequencyData(d);v=d.reduce((a,b)=>a+b,0)/d.length/110}
  else if(!muted){v=level}
  level*=0.9;
  if(!stage.classList.contains('thinking'))orb.style.transform=`scale(${1+Math.min(v,0.6)*0.5})`;
  requestAnimationFrame(loop);
}
$('start').onclick=start;$('stop').onclick=stop;
$('mute').onclick=()=>{muted=!muted;$('mute').textContent=muted?'إلغاء الكتم':'كتم الميكروفون'};
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

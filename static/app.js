(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const EXAMPLES = [
    { en: 'butterfly', zh: '蝴蝶' },
    { en: 'flower', zh: '花' },
    { en: 'garden', zh: '花园' },
    { en: 'sunny', zh: '阳光明媚的' },
    { en: 'friend', zh: '朋友' },
  ];
  const date = new Date();
  const today = [date.getFullYear(), String(date.getMonth()+1).padStart(2,'0'), String(date.getDate()).padStart(2,'0')].join('-');
  const wordsKey = `reading-buddy:words:${today}`;
  const progressKey = `reading-buddy:progress:${today}`;
  const getStored = (key) => { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } };
  const setStored = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} };
  const loaded = getStored(wordsKey);
  let words = Array.isArray(loaded) && loaded.length ? loaded : EXAMPLES.map(w=>({...w}));
  let passed = getStored(progressKey);
  passed = Array.isArray(passed) && passed.length === words.length ? passed.map(Boolean) : words.map(()=>false);
  let index = Math.max(0, passed.findIndex(x=>!x));
  if(passed.every(Boolean)) index = words.length;
  let recorder = null;
  let recordingStarted = 0;
  let timer = null;
  let stream = null;
  let busy = false;

  $('todayDate').textContent = `${date.toLocaleDateString('zh-CN', {month:'long',day:'numeric',weekday:'long'})} · 每天 5 分钟，勇敢开口读！`;
  $('wordEditor').value = words.map(w => `${w.en} | ${w.zh}`).join('\n');

  function setResult(kind, title, detail, heard='') {
    const box = $('resultBox');
    box.className = `result-box ${kind}`;
    box.replaceChildren();
    const heading = document.createElement('strong');
    heading.textContent = title;
    const message = document.createElement('div');
    message.textContent = detail;
    box.append(heading, message);
    if (heard) {
      const line = document.createElement('div');
      line.className='heard';
      line.textContent = `AI 听到的是：${heard}`;
      box.append(line);
    }
  }
  function hideFeedback() {
    $('resultBox').classList.add('hidden');
    $('nextBtn').classList.add('hidden');
    $('retryBtn').classList.add('hidden');
  }
  function persist() { setStored(progressKey, passed); }
  function show() {
    const done = passed.filter(Boolean).length;
    $('countLabel').textContent = `今日任务 ${done} / ${words.length}`;
    $('progressDots').replaceChildren();
    words.forEach((word, i) => {
      const d = document.createElement('div');
      d.className=`dot ${passed[i]?'done':i===index?'current':''}`;
      d.title = `${i+1}. ${word.en}${passed[i]?'（已完成）':''}`;
      $('progressDots').append(d);
    });
    const finished = index >= words.length;
    $('finishView').classList.toggle('hidden', !finished);
    $('practiceView').classList.toggle('hidden', finished);
    if(finished) return;
    const word=words[index];
    $('wordPosition').textContent=`WORD ${String(index+1).padStart(2,'0')}`;
    $('wordText').textContent=word.en;
    $('wordMeaning').textContent=word.zh||'勇敢读出来就好！';
    hideFeedback();
    if (passed[index]) {
      setResult('', '🌟 这个词已经完成啦！', '可以去读下一张单词卡。');
      $('nextBtn').classList.remove('hidden');
    }
  }
  function next() {
    const i = passed.findIndex((v, j)=>!v && j>index);
    if(i!==-1) index=i;
    else index=passed.findIndex(x=>!x);
    if(index===-1) index=words.length;
    show();
  }
  function success() {
    passed[index]=true;
    persist();
    setResult('', '🌟 AI 听到了！', '你念出了这个单词，真棒！');
    $('retryBtn').classList.add('hidden');
    $('nextBtn').classList.remove('hidden');
    const done=passed.filter(Boolean).length;
    $('countLabel').textContent=`今日任务 ${done} / ${words.length}`;
    const dots=$('progressDots').children;
    if(dots[index]) dots[index].className='dot done';
  }
  function speak() {
    if (!('speechSynthesis' in window)) return setResult('error', '当前浏览器没有朗读功能', '请使用 Chrome、Edge 或 Safari。');
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(words[index].en);
    utter.lang='en-US'; utter.rate=.78; utter.pitch=1.04;
    const voices=window.speechSynthesis.getVoices();
    const english=voices.find(v=>v.lang.toLowerCase().startsWith('en'));
    if(english) utter.voice=english;
    window.speechSynthesis.speak(utter);
  }
  const mimeCandidates = ['audio/webm;codecs=opus','audio/webm','audio/mp4','audio/ogg;codecs=opus'];
  function setRecordUI(isRecording) {
    $('recordBtn').classList.toggle('recording',isRecording);
    $('recordLabel').textContent=isRecording?'结束朗读':'开始朗读';
    $('recordNote').textContent=isRecording?'正在录音，再点一次结束，最多 8 秒':'点击开始，再点一次结束（最长 8 秒）';
  }
  function closeMic() {
    if(timer){clearTimeout(timer);timer=null;}
    if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
    setRecordUI(false);
  }
  async function sendAudio(audio, mime, wordAtStart) {
    busy=true; $('recordBtn').disabled=true;
    setResult('', '🔎 本地 AI 正在听…', '第一次使用需要下载语音模型，请不要关闭网页。');
    try {
      const res=await fetch('/api/recognize', {
        method:'POST', headers: {'Content-Type':mime,'X-Reading-Buddy':'local-only'}, body:audio,
      });
      const data=await res.json();
      if(!res.ok) throw new Error(data.error || '无法识别，请稍后再试');
      if(!words[index] || words[index].en!==wordAtStart) return;
      const heard=String(data.heard||'').trim();
      const normalize=t=>t.toLowerCase().replace(/[’]/g,"'").replace(/[^a-z0-9'\s-]/g,' ').replace(/-/g,' ').trim().replace(/\s+/g,' ');
      if(heard && normalize(heard) === normalize(wordAtStart)) {
        success();
      } else {
        setResult('retry', '🌼 我还没有完全听清', '这不代表发音错了。先听示范，再慢慢读一遍吧！', heard);
        $('retryBtn').classList.remove('hidden');
      }
    } catch(err) {
      setResult('error','暂时听不到你的声音', err.message||'本地 AI 连接失败，请让家长检查。');
      $('retryBtn').classList.remove('hidden');
    } finally { busy=false;$('recordBtn').disabled=false; }
  }
  async function startRecord() {
    if(busy || index>=words.length || passed[index]) return;
    hideFeedback();
    if(!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      return setResult('error','暂时不能录音','请在电脑上的 Chrome/Edge 浏览器通过 http://localhost:8765 打开。');
    }
    try {
      stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true},video:false});
      const mime=mimeCandidates.find(m=>MediaRecorder.isTypeSupported(m));
      recorder=new MediaRecorder(stream,mime?{mimeType:mime}:undefined);
      const parts=[];
      const wordAtStart=words[index].en;
      recorder.ondataavailable=e=>{if(e.data.size)parts.push(e.data)};
      recorder.onstop=async()=>{
        const time=performance.now()-recordingStarted;
        const actualMime=recorder?.mimeType||mime||'audio/webm';
        closeMic();
        if(time<650){return setResult('retry','录得有点短','请完整地读一遍，再结束录音。');}
        if(!parts.length){return setResult('error','没有录到声音','请检查麦克风权限。');}
        await sendAudio(new Blob(parts,{type:actualMime}),actualMime,wordAtStart);
      };
      recorder.onerror=()=>{closeMic();setResult('error','录音失败','请检查麦克风权限。')};
      recorder.start(); recordingStarted=performance.now(); setRecordUI(true);
      timer=setTimeout(()=>stopRecord(),8000);
    } catch(err) {
      closeMic();
      setResult('error','无法打开麦克风','请允许网页使用麦克风，并检查是否被其他程序占用。');
    }
  }
  function stopRecord() { if(recorder?.state==='recording') recorder.stop(); }
  $('recordBtn').addEventListener('click',()=>{
    if(recorder?.state==='recording')stopRecord();
    else startRecord();
  });
  $('speakBtn').addEventListener('click',speak);
  $('nextBtn').addEventListener('click',next);
  $('retryBtn').addEventListener('click',()=>{hideFeedback();$('recordBtn').focus();});
  $('repeatBtn').addEventListener('click',()=>{index=0;passed=words.map(()=>false);persist();show();});
  $('parentToggle').addEventListener('click',()=>{
    const panel=$('parentPanel');
    panel.classList.toggle('hidden');
    if(!panel.classList.contains('hidden'))panel.scrollIntoView({behavior:'smooth',block:'nearest'});
  });
  $('saveWords').addEventListener('click',()=>{
    if(busy || recorder?.state==='recording')return;
    const lines=$('wordEditor').value.split(/\r?\n/).map(x=>x.trim()).filter(Boolean);
    if(lines.length===0 || lines.length>30){$('editInfo').textContent='请输入 1–30 行单词。';return;}
    const nextWords=[];
    for (const line of lines) {
      const [en,...rest]=line.split(/[|｜]/);
      const text=(en||'').trim();
      if(text.length<1 || text.length>70 || !/[a-z]/i.test(text) || /[^a-zA-Z0-9\s'’\-!?.,]/.test(text)){
        $('editInfo').textContent=`请检查英文内容：${line.slice(0,40)}`;return;
      }
      nextWords.push({en:text,zh:rest.join('|').trim().slice(0,60)});
    }
    words=nextWords;passed=words.map(()=>false);index=0;
    setStored(wordsKey,words);persist();show();
    $('editInfo').textContent=`已保存 ${words.length} 个单词／短句，今日进度已重新开始。`;
  });
  $('resetProgress').addEventListener('click',()=>{
    if(!window.confirm('清除今天的朗读进度？单词表不会丢失。'))return;
    passed=words.map(()=>false);index=0;persist();show();
    $('editInfo').textContent='今天的闯关进度已清除。';
  });
  $('parentAdvance').addEventListener('click',()=>{
    if(index>=words.length || busy)return;
    success();
    $('editInfo').textContent='已由家长确认本词完成。';
  });
  fetch('/api/health').then(r=>r.json()).then(data=>{
    if(!data.aiInstalled){$('aiState').textContent='⚠️ AI 待安装';$('aiState').title='运行 pip install -r requirements.txt';}
    else {$('aiState').textContent='🔒 本地 AI 已连接';$('aiState').title='首次使用时模型会下载到本地';}
  }).catch(()=>{$('aiState').textContent='⚠️ 服务未连接'});
  show();
})();

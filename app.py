import os, tempfile, json, re, html
from pathlib import Path
import cv2, streamlit as st, networkx as nx, pandas as pd, requests
from pyvis.network import Network
import streamlit.components.v1 as components
from ultralytics import YOLO
import factory_core as core

st.set_page_config(page_title='Smart Factory Ontology',layout='wide')
st.title('Smart Factory Ontology')
st.caption('YOLOv8 · 프레임별 지식그래프 · 누적 지식그래프 · LLM')
MODEL=Path(__file__).parent/'models'/'best.pt'

def graph_view(G, height=520):
    net=Network(height=f'{height}px',width='100%',directed=True,bgcolor='#101b2d',font_color='white',cdn_resources='in_line')
    colors={'instance':'#38bdf8','class':'#a78bfa','process':'#34d399','frame':'#fbbf24'}
    for n,d in G.nodes(data=True):
        net.add_node(str(n),label=str(n),title=html.escape(json.dumps(d,ensure_ascii=False,default=str)),color=colors.get(d.get('kind'),'#94a3b8'))
    for a,b,d in G.edges(data=True):
        candidate=d.get('evidence')=='rule_candidate'
        net.add_edge(str(a),str(b),label=d.get('label',''),arrows='to',dashes=candidate,color='#f59e0b' if candidate else '#94a3b8')
    net.set_options(json.dumps({'interaction':{'hover':True,'navigationButtons':False,'dragView':True,'zoomView':True},'physics':{'stabilization':{'iterations':80},'barnesHut':{'springLength':150}}}))
    toolbar='''<div style="padding:8px;text-align:right;background:#101b2d"><button onclick="zoom(1.25)">＋ 확대</button> <button onclick="zoom(0.8)">－ 축소</button> <button onclick="network.fit()">전체 보기</button></div><script>function zoom(f){network.moveTo({scale:network.getScale()*f});}</script>'''
    page=net.generate_html(); page=re.sub(r'(<body[^>]*>)',lambda m:m.group(1)+toolbar,page,count=1)
    components.html(page,height=height+100,scrolling=False)

with st.sidebar:
    st.header('사용방법')
    with st.container(border=True):
        st.markdown("""**영상 분석 순서**

1. 영상 파일을 업로드합니다.
2. 아래 분석 설정을 조절합니다.
3. 영상 분석 시작 버튼을 누릅니다.
4. 레이블링 확인 탭에서 객체를 확인합니다.
5. 지식그래프 탭에서 객체와 관계를 확인합니다.
6. LLM 탭에서 질문을 입력합니다.

**분석 설정 설명**

- **프레임 분석 간격(초):** 몇 초마다 영상의 한 장면을 분석할지 설정합니다.
- **YOLO 탐지 신뢰도:** 설정한 점수 이상의 객체만 탐지 결과에 포함합니다.
- **최대 분석 시점 수:** 분석할 영상 장면의 최대 개수를 설정합니다.
""")
    st.divider()
    upload=st.file_uploader('영상 업로드',type=['mp4','mov','avi','mkv'])
    interval=st.slider('프레임 분석 간격(초)',1,10,2)
    conf=st.slider('YOLO 탐지 신뢰도',0.1,0.9,0.25,0.05)
    max_steps=st.slider('최대 분석 시점 수',10,300,80,10)
    timeline=st.checkbox('기존 펜팩토리 영상 시간표 적용',False)
    start=st.button('영상 분석 시작',disabled=upload is None,type='primary')

if start:
    st.session_state.pop('result',None); st.session_state.pop('answer',None)
    path=None; cap=None
    try:
        with tempfile.NamedTemporaryFile(delete=False,suffix=Path(upload.name).suffix) as f:
            f.write(upload.getbuffer()); path=f.name
        model=YOLO(str(MODEL)); cap=cv2.VideoCapture(path)
        if not cap.isOpened(): raise ValueError('영상을 열 수 없습니다.')
        fps=cap.get(cv2.CAP_PROP_FPS) or 30
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); diag=max(1,(cap.get(3)**2+cap.get(4)**2)**.5)
        G=nx.MultiDiGraph(); tracker=core.Tracker(); snaps=[]; prev=None; prev_proc=None
        for c,p in core.PARENT.items():
            G.add_node(c,kind='class'); G.add_node(p,kind='class'); G.add_edge(c,p,label='subClassOf')
        steps=min(max_steps,max(1,int((count-1)/fps/interval)+1)); progress=st.progress(0); live=st.empty()
        for k in range(steps):
            t=k*interval; cap.set(cv2.CAP_PROP_POS_FRAMES,int(t*fps)); ok,frame=cap.read()
            if not ok: break
            dets=core.detect_yolo(frame,model,conf); tracker.update(dets,diag)
            active={d['id']:{'cls':d['cls'],'status':'present'} for d in dets}
            seen={d['id']:{'cls':d['cls'],'c':core.center(d['box']),'t':t} for d in dets}
            spatial=core.infer_spatial(dets,diag); semantic=core.infer_semantic(active,seen)
            rels=list(dict.fromkeys(spatial+semantic)); proc=core.process_at(t) if timeline else 'UnknownProcess'
            S=core.build_snapshot(t,active,rels,proc,prev_proc); prev_proc=proc
            prev=core.update_global(k,t,dets,rels,proc,prev,graph=G)
            rows=[]
            for d in dets:
                x1,y1,x2,y2=map(int,d['box']); cv2.rectangle(frame,(x1,y1),(x2,y2),(40,220,180),2)
                cv2.putText(frame,f"{d['raw_cls']} {d['id']} {d['conf']:.2f}",(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,220,180),2)
                rows.append({'id':d['id'],'label':d['raw_cls'],'class':d['cls'],'confidence':round(d['conf'],3),'bbox':[x1,y1,x2,y2]})
            scale=min(1,960/frame.shape[1]); preview=cv2.resize(frame,None,fx=scale,fy=scale); _,jpg=cv2.imencode('.jpg',preview)
            snaps.append({'time':t,'graph':S,'rows':rows,'relations':rels,'image':jpg.tobytes(),'process':proc})
            progress.progress((k+1)/steps); live.image(jpg.tobytes(),caption=f'{t}초 · {len(rows)}개 탐지')
        if not snaps: raise ValueError('분석 가능한 프레임이 없습니다.')
        st.session_state.result=(snaps,G); live.empty(); st.success(f'{len(snaps)}개 시점 분석 완료')
    except Exception as e: st.error(f'분석 오류: {e}')
    finally:
        if cap is not None: cap.release()
        if path: os.unlink(path)

if 'result' not in st.session_state:
    st.info('왼쪽에서 영상을 업로드하고 분석을 시작하세요.'); st.stop()
snaps,G=st.session_state.result
idx=st.select_slider('공통 분석 시점 (초)',options=list(range(len(snaps))),format_func=lambda i:str(snaps[i]['time']),value=len(snaps)-1)
s=snaps[idx]; ids={r['id'] for r in s['rows']}; nodes={n for n,d in s['graph'].nodes(data=True) if d.get('kind')=='instance'}
llm,kg,labels=st.tabs(['LLM','지식그래프','레이블링 확인'])
with labels:
    st.image(s['image'],caption=f"{s['time']}초 · YOLO 탐지 결과")
    st.dataframe(pd.DataFrame(s['rows']))
    st.write('그래프에 누락된 탐지 ID:',sorted(ids-nodes)); st.write('탐지에 없는 그래프 객체:',sorted(nodes-ids))
    st.metric('탐지 → 그래프 반영률',f'{100*len(ids&nodes)/len(ids):.1f}%' if ids else '탐지 없음')
    st.caption('반영률은 모델의 탐지 정확도(mAP)가 아닙니다. 실제 영상과 박스를 비교해 검증하세요.')
with kg:
    st.subheader('Temporary 지식그래프'); graph_view(s['graph']); st.dataframe(pd.DataFrame(s['relations'],columns=['주체','관계 후보','대상']))
    st.caption('주황 점선은 위치 또는 온톨로지 규칙 기반 추론 후보이며 실제 작동을 확정하지 않습니다.')
    st.subheader('누적 지식그래프')
    st.caption(f'{G.number_of_nodes()}개 노드 · {G.number_of_edges()}개 관계')
    limit=st.slider('표시 노드 수',1,max(2,G.number_of_nodes()),min(160,G.number_of_nodes()))
    order=sorted(G,key=lambda n: {'instance':0,'process':1,'class':2,'frame':3}.get(G.nodes[n].get('kind'),4)); graph_view(G.subgraph(order[:limit]))
    st.download_button('전체 그래프 JSON',json.dumps(nx.node_link_data(G),ensure_ascii=False,default=str),'graph.json')
with llm:
    st.subheader('LLM Prompt · Ollama / Qwen')
    def setting(key,default):
        try: return st.secrets.get(key,os.getenv(key,default))
        except Exception: return os.getenv(key,default)
    base=setting('OLLAMA_URL','http://127.0.0.1:11434').rstrip('/'); name=setting('OLLAMA_MODEL','qwen2.5:3b'); token=setting('OLLAMA_TOKEN','')
    headers={'Authorization':f'Bearer {token}'} if token else {}
    st.caption(f'모델: {name} · 공개 배포 시 서버에서 접근 가능한 인증된 Ollama 주소 필요')
    if st.button('LLM 연결 확인'):
        try:
            r=requests.get(base+'/api/tags',headers=headers,timeout=8); r.raise_for_status(); names=[m['name'] for m in r.json()['models']]; st.success('연결됨: '+', '.join(names)) if name in names else st.warning('연결됨, 지정 모델 없음: '+', '.join(names))
        except Exception: st.error('Ollama 연결 실패: 서버 주소, 인증, 실행 상태 확인')
    scope=st.radio('질문 범위',['선택 시점','전체 분석 시점'],horizontal=True)
    q=st.text_area('질문',placeholder='로봇팔과 컨베이어의 관계를 설명해줘')
    if st.button('질문하기',disabled=not q.strip()):
        selected=[s] if scope=='선택 시점' else snaps
        evidence=[{'time':x['time'],'objects':x['rows'],'relations_candidates':x['relations'],'process':x['process']} for x in selected]
        # Bound prompt size; do not silently omit evidence.
        payload=json.dumps(evidence,ensure_ascii=False)
        if len(payload)>45000: st.warning('전체 데이터가 너무 큽니다. 선택 시점으로 질문하거나 분석 시점 수를 줄여주세요.')
        else:
            try:
                with st.spinner('LLM 추론 중'):
                    r=requests.post(base+'/api/chat',headers=headers,json={'model':name,'stream':False,'messages':[{'role':'system','content':'한국어로 답하고 근거 시각과 객체 ID를 표시하라. 탐지 결과와 관계 후보를 구분하고 실제 제어를 단정하지 마라. 데이터에 없으면 모른다고 답하라.'},{'role':'user','content':q+'\n분석 데이터:\n'+payload}]},timeout=180); r.raise_for_status(); st.session_state.answer=r.json()['message']['content']
            except Exception: st.error('LLM 응답 실패. Ollama 연결 및 모델 상태를 확인하세요.')
    if 'answer' in st.session_state: st.write(st.session_state.answer)

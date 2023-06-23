import os, json, requests
from dotenv import load_dotenv
from oracleDB import oracleDB
from imageModelAI import ImageAI
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from starlette.responses import RedirectResponse

load_dotenv()
app = FastAPI()
db = oracleDB(os.environ.get('ID'), os.environ.get('PW'), os.environ.get('IP'), os.environ.get('PORT'), os.environ.get('SID'))

@app.get('/favicon.ico', include_in_schema=False)
async def favicon() -> None:
    return FileResponse('main.py')

def get_datas(project_no: int) -> tuple:
    db.learning_start(project_no)
    data_bundles = db.data(project_no)[0]
    labeling_dones = db.labeling(project_no)
    labels = json.loads(db.label_names(project_no)[0])['info'] # str을 json으로 바꾸고 'info'로 접속 -> [라벨명, 라벨명, ...]이 return
    return (data_bundles, labeling_dones, labels)

async def learn_ai(project_no: int) -> None:
    try:
        (data_bundles, labeling_dones, labels) = get_datas(project_no)
        if not data_bundles or not labeling_dones:
            print({'no': 'data'})

        ai = ImageAI()
        (array, originals) = await ai.save_img(project_no, data_bundles, labeling_dones) # [image파일명, 작업자, 고른 답]을 return
        convert_images_3D = await ai.convert_to_num(project_no, data_bundles) # 이미지를 3차원 배열로 변환
        ai.color(project_no, array, originals, convert_images_3D, labels)
        db.learning_end(project_no)
    except:
        db.learning_rollback(project_no)
        print({'error': 'occurred'})

@app.get("/labeling.result/{project_no}", status_code=202)
async def select(project_no: int, bts: BackgroundTasks) -> str:
    bts.add_task(learn_ai, project_no=project_no)
    return 'accepted'
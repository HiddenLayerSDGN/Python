from cx_Oracle import connect

class DataBundle:
    def __init__(self, bundle_no: int, bundle_uploader: str, bundle_uploaded_filename: str, bundle_data_type: str, bundle_folder_name: str) -> None:
        self.bundle_no = bundle_no # 이미지 url에 쓰임
        self.bundle_uploader = bundle_uploader # 이미지 저장 폴더 경로에 쓰일 예정 (사실 필수는 아니나, 내 마음임)
        self.bundle_uploaded_filename = bundle_uploaded_filename # 이미지 저장 폴더 경로에 쓰일 예정 (사실 필수는 아니나, 내 마음임)
        self.bundle_data_type = bundle_data_type # jpg인지 png인지 구분
        self.bundle_folder_name = bundle_folder_name # 이미지 url에 쓰임

class Labeling_Done:
    def __init__(self, data_no: str, worked_by: str, label: str) -> None:
        self.data_no = data_no # 이름은 data_no인데, 이거 data_name임! 이미지 url에 필요함, 마지막 주소 부분
        self.worked_by = worked_by # 라벨링 맡은 사람
        self.label = label # 맡은 사람이 누른 답

class oracleDB:
    def __init__(self, id, pw, ip, port, sid) -> None:
        self.id = id
        self.pw = pw
        self.ip = ip
        self.port = port
        self.sid = sid
        self.con = connect(f'{self.id}/{self.pw}@{self.ip}:{self.port}/{self.sid}')
        self.cur = self.con.cursor()

    def execute(self, sql: str) -> None:
        self.cur.execute(sql)

    def learning_start(self, project_no: int) -> None:
        sql = f"update Labeling_Project set project_progress = 2 where project_no = {project_no}"
        self.execute(sql)
        self.con.commit()

    def learning_end(self, project_no: int) -> None:
        sql = f"update Labeling_Project set project_progress = 3 where project_no = {project_no}"
        self.execute(sql)
        self.con.commit()

    def learning_rollback(self, project_no: int) -> None:
        sql = f"update Labeling_Project set project_progress = 1 where project_no = {project_no}"
        self.execute(sql)
        self.con.commit()

    def data(self, project_no: int) -> list:
        sql = f"select bundle_no, bundle_uploader, bundle_uploaded_filename, bundle_data_type, bundle_folder_name from DataBundle where bundle_no = (select project_bundle_no from Labeling_Project where project_no = {project_no}) order by bundle_no"
        self.execute(sql)
        return self.get_DataBundles()

    def labeling(self, project_no: int) -> list:
        sql = f"select data_no, worked_by, label from Labeling_Done where project_no = {project_no} order by project_no"    
        self.execute(sql)
        return self.get_Labeling_Done()

    def label_names(self, project_no: int) -> list:
        sql = f"select project_category from Labeling_Project where project_no = {project_no} order by project_no"
        self.execute(sql)
        return self.get_label_names()
    
    def get_DataBundles(self) -> list:
        return [DataBundle(bundle_no, bundle_uploader, bundle_uploaded_filename, bundle_data_type, bundle_folder_name) for bundle_no, bundle_uploader, bundle_uploaded_filename, bundle_data_type, bundle_folder_name in self.cur]

    def get_Labeling_Done(self) -> list:
        return [Labeling_Done(data_no, worked_by, label) for data_no, worked_by, label in self.cur]
    
    def get_label_names(self) -> list:
        return self.cur.fetchone()
    
    def insert_result(self, project_no: int, images: list, answers: list) -> None:
        for img, ans in zip(images, answers):
            sql = f"insert into Labeling_Result values({project_no}, {img}, '{ans}')"
            self.execute(sql)
            self.con.commit()

    def update_trust(self, score_list: list) -> None:
        for score, labeler in score_list:
            sql = f"update Member set user_trust = {score} where user_id = '{labeler}'"
            self.execute(sql)
            self.con.commit()
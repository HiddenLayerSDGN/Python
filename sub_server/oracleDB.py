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

# url = bundle_no + bundle_folder_name + data_no 으로 구성되어 있다, save_img 메소드 참고

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

    def get_DataBundles(self) -> list:
        return [DataBundle(bundle_no, bundle_uploader, bundle_uploaded_filename, bundle_data_type, bundle_folder_name) for bundle_no, bundle_uploader, bundle_uploaded_filename, bundle_data_type, bundle_folder_name in self.cur]

    def get_Labeling_Done(self) -> list:
        return [Labeling_Done(data_no, worked_by, label) for data_no, worked_by, label in self.cur]
    
    def get_label_counts(self) -> int:
        return self.cur.fetchone()
from dotenv import load_dotenv
from src.db import WalkieVectorDB, ObjectRecord, SceneRecord, PersonRecordb
load_dotenv()

def main():
    db = WalkieVectorDB()
    db.add_object(ObjectRecord(id="1", name="Object 1", description="Object 1 description"))
    db.add_scene(SceneRecord(id="1", name="Scene 1", description="Scene 1 description"))
    db.add_person(PersonRecord(id="1", name="Person 1", description="Person 1 description"))
    move_absolute.invoke({"x":1, "y":1, "heading":0.0})
    # move_relative.invoke({"x":1, "y":0, "heading":0.0})


if __name__ == "__main__":
    main()

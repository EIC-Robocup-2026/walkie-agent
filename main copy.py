from dotenv import load_dotenv
from src.vision import print_cameras
load_dotenv()

def main():
    print_cameras()
    # move_absolute.invoke({"x":1, "y":1, "heading":0.0})
    # move_relative.invoke({"x":1, "y":0, "heading":0.0})


if __name__ == "__main__":
    main()

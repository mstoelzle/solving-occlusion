import sys

from src.utils.sheet_uploader.sheet_uploader import SheetUploader

if __name__ == '__main__':
    exp_name = sys.argv[1]
    print(f"manually uploading experiment {exp_name}")
    uploader = SheetUploader(exp_name)
    uploader.upload()

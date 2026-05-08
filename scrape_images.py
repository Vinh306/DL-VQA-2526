from icrawler.builtin import BingImageCrawler
from PIL import Image
import os, time, sys


TARGET_SIZE = (384, 384)
TARGET_PER_FOOD = 30
FINAL_DIR = "data/images"

FOODS = [
    "bánh mì Việt Nam", "phở bò", "bún bò Huế", "cơm tấm",
    "bánh xèo", "gỏi cuốn", "bún chả", "hủ tiếu",
    "bánh cuốn", "cháo lòng", "mì Quảng", "bún riêu",
    "bánh canh", "lẩu Thái", "cơm gà Hội An"
]


# Hàm resize ảnh
def resize_img(folder, size=TARGET_SIZE):
    images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    ok = 0
    for img_file in images:
        path = os.path.join(folder, img_file)
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            if w < 100 or h < 100:
                os.remove(path)
                continue
            img = img.resize(size, Image.LANCZOS)
            jpg_path = os.path.splitext(path)[0] + ".jpg"
            img.save(jpg_path, format="JPEG", quality=90)
            if path != jpg_path:
                os.remove(path)
            ok += 1
        except:
            if os.path.exists(path):
                os.remove(path)
    return ok

# hàm lấy ảnh
def run_new():
    print("Lấy ảnh mới từ đầu")
    print(f"Mục tiêu: {TARGET_PER_FOOD} ảnh/món")
    os.makedirs(FINAL_DIR, exist_ok=True)
    for food in FOODS:
        food_slug    = food.replace(" ", "_")
        final_folder = os.path.join(FINAL_DIR, food_slug)
        os.makedirs(final_folder, exist_ok=True)
        existing = [f for f in os.listdir(final_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        # Bỏ qua nếu đã đủ ảnh
        if len(existing) >= TARGET_PER_FOOD:
            print(f"[{food}]: đã đủ {len(existing)} ảnh, bỏ qua\n")
            continue
        crawl_num = TARGET_PER_FOOD * 2
        print(f"Scraping [{food}] — {crawl_num} ảnh...")
        crawler = BingImageCrawler(storage={"root_dir": final_folder})
        crawler.crawl(keyword=food, max_num=crawl_num, file_idx_offset=0)
        saved = resize_img(final_folder)
        print(f"[{food}]: {saved} ảnh, resize {TARGET_SIZE}\n")
        time.sleep(2)


if __name__ == "__main__":
    run_new()
    print(f"Hoàn thành. Ảnh lưu tại: {FINAL_DIR}")
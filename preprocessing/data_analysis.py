import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def generate_metadata_and_eda(dataset_root):
    """
    Quét thư mục, tạo metadata, kiểm tra leakage và vẽ biểu đồ thống kê
    """
    print("Tạo metadata")
    data = []
    
    # Quét toàn bộ ảnh trong các thư mục train/val/test
    all_images = glob.glob(os.path.join(dataset_root, '*', '*', '*.jpg'))
    
    for img_path in all_images:
      
        parts = img_path.split(os.sep)
        split = parts[-3]  # train, val, test
        label = parts[-2]  # real, fake
        filename = parts[-1]
        
        # Trích xuất video_id 
        video_id = filename.split('_f')[0]
        
        data.append({
            'image_path': img_path,
            'split': split,
            'label': label,
            'video_id': video_id,
            'filename': filename
        })
        
    df = pd.DataFrame(data)
    
    # Lưu ra file CSV
    csv_path = os.path.join(dataset_root, 'metadata.csv')
    df.to_csv(csv_path, index=False)
    print(f"Đã tạo xong {csv_path} với {len(df)} dòng")
    
    print("\nKiểm tra data leakage")
   
    train_vids = set(df[df['split'] == 'train']['video_id'])
    val_vids = set(df[df['split'] == 'val']['video_id'])
    test_vids = set(df[df['split'] == 'test']['video_id'])
    

    leak_train_val = train_vids.intersection(val_vids)
    leak_train_test = train_vids.intersection(test_vids)
    leak_val_test = val_vids.intersection(test_vids)
    
    if not leak_train_val and not leak_train_test and not leak_val_test:
        print("Không bị leakage")
    else:
        print("Phát hiện leakage:")
        if leak_train_val: print(f" - Trùng giữa Train và Val: {len(leak_train_val)} video.")
        if leak_train_test: print(f" - Trùng giữa Train và Test: {len(leak_train_test)} video.")
        if leak_val_test: print(f" - Trùng giữa Val và Test: {len(leak_val_test)} video.")
        
    print("\nThống kê số lượng ảnh:")
    # Đếm số lượng ảnh theo Split và Label
    stats = df.groupby(['split', 'label']).size().unstack(fill_value=0)
    print(stats)
    
    # biểu đồ Bar Chart
    stats.plot(kind='bar', stacked=False, figsize=(8, 6), colormap='viridis')
    plt.title('Thống kê số lượng ảnh Real/Fake trong các tập')
    plt.xlabel('Tập dữ liệu')
    plt.ylabel('Số lượng ảnh')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
   
    print("\nĐang hiển thị biểu đồ thống kê")
    plt.show()

if __name__ == "__main__":
    OUTPUT_ROOT_DIR = r"D:\python\output_dataset"
    generate_metadata_and_eda(OUTPUT_ROOT_DIR)
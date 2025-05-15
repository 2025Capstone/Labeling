import zstandard as zstd
import os
import csv
import pandas as pd

# 압축된 zstd 파일(.csv.zst)을 해제하여 원본 CSV로 복구하는 스크립트
# 사용법: python decode_zstd_csv.py data/label1.csv.zst data/label1_wearable.csv data/video1_drowsiness_label.csv data/label1_merged.csv

def decompress_zstd_csv(zstd_path, wearable_path, drowsiness_path, out_csv_path=None):
    if not zstd_path.endswith('.zst'):
        raise ValueError('입력 파일은 .zst 확장자를 가져야 합니다.')
    for p in (wearable_path, drowsiness_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f'CSV 파일을 찾을 수 없습니다: {p}')        

    if out_csv_path is None:
        out_csv_path = zstd_path[:-4]  # .zst 제거
    
    tmp_csv = out_csv_path + '.tmp'
    dctx = zstd.ZstdDecompressor()
    with open(zstd_path, 'rb') as f_in, open(tmp_csv, 'w', encoding='utf-8', newline='') as f_out:
        with dctx.stream_reader(f_in) as reader:
            # Read and write in chunks for memory efficiency
            while True:
                chunk = reader.read(16384)  # 16KB
                if not chunk:
                    break
                f_out.write(chunk.decode('utf-8'))
    print(f"랜드마크 복호화 완료: {tmp_csv}")
    
    # pandas로 불러와 병합
    df_land = pd.read_csv(tmp_csv)
    df_wear = pd.read_csv(wearable_path)
    df_drow = pd.read_csv(drowsiness_path)
    
    intervals = pd.IntervalIndex.from_arrays(
        df_wear['Timestamp'],
        df_wear['Segment End'],
        closed='both'
    )
    
    # 웨어러블 피처만 뽑아놓고
    wear_feats = df_wear.drop(columns=['Timestamp','Segment End'])
    
    # 각 랜드마크 타임스탬프에 대응하는 세그먼트 인덱스 찾기
    idx = intervals.get_indexer(df_land['Timestamp'])

    # 인덱스를 통해 피처 할당 (일치하지 않으면 -1 → NaN)
    df_wear_aligned = wear_feats.reindex(idx).reset_index(drop=True)
        
    # 합치기
    df_merged = pd.concat([df_land.reset_index(drop=True), df_wear_aligned], axis=1)
    
    df_merged['drowsiness'] = pd.NA
    
    for _, label_row in df_drow.iterrows():
        start = float(label_row['start_time_sec'])
        end = float(label_row['end_time_sec'])
        label = label_row['label']
        mask = (df_merged['Timestamp'] >= start) & (df_merged['Timestamp'] <= end)
        df_merged.loc[mask, 'drowsiness'] = label
    
    # 최종 파일로 저장
    df_merged.to_csv(out_csv_path, index=False, encoding='utf-8')
    print(f"✅ 통합 CSV 저장 완료: {out_csv_path}")
    
    # 임시 파일 정리
    os.remove(tmp_csv)
    
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("사용법: python decode_zstd_csv.py <압축된_csv.zst> <wearable_label.csv> <drowsiness_label.csv> [<출력_csv>]")
        exit(1)
    zstd_path = sys.argv[1]
    wearable_path = sys.argv[2]
    drowsiness_path = sys.argv[3]
    out_csv = sys.argv[4] if len(sys.argv) >= 5 else None
    
    decompress_zstd_csv(zstd_path, wearable_path, drowsiness_path, out_csv)

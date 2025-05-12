import zstandard as zstd
import os
import csv
import pandas as pd

# 압축된 zstd 파일(.csv.zst)을 해제하여 원본 CSV로 복구하는 스크립트
# 사용법: python decode_zstd_csv.py data/label1.csv.zst

def decompress_zstd_csv(zstd_path, wearable_path, out_csv_path=None):
    if not zstd_path.endswith('.zst'):
        raise ValueError('입력 파일은 .zst 확장자를 가져야 합니다.')
    if not os.path.exists(wearable_path):
        raise FileNotFoundError(f'웨어러블 CSV를 찾을 수 없습니다: {wearable_path}')
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
    print(f"복호화 완료: {tmp_csv}")
    
    # 3) pandas로 불러와 병합
    df_land = pd.read_csv(tmp_csv)
    df_wear = pd.read_csv(wearable_path)
    # Segment End 컬럼 제거
    if 'Segment End' in df_wear.columns:
        df_wear = df_wear.drop(columns=['Segment End'])
        
    # Round wearable timestamps to nearest 2-minute multiple
    df_wear['Timestamp'] = (df_wear['Timestamp'] / 120.0).round() * 120.0
    
    # Timestamp 기준 left join
    df_merged = pd.merge(df_land, df_wear, on='Timestamp', how='left')
    
    # 4) 최종 파일로 저장
    df_merged.to_csv(out_csv_path, index=False, encoding='utf-8')
    print(f"✅ 통합 CSV 저장 완료: {out_csv_path}")
    
    # 5) 임시 파일 정리
    os.remove(tmp_csv)
    
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("사용법: python decode_zstd_csv.py <압축된_csv.zst> <wearable_label.csv> [<출력_csv>]")
        exit(1)
    zstd_path = sys.argv[1]
    wearable_path = sys.argv[2]
    out_csv = sys.argv[3] if len(sys.argv) >= 4 else None
    
    decompress_zstd_csv(zstd_path, wearable_path, out_csv)

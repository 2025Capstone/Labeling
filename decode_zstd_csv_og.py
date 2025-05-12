import zstandard as zstd
import os
import csv

# 압축된 zstd 파일(.csv.zst)을 해제하여 원본 CSV로 복구하는 스크립트
# 사용법: python decode_zstd_csv.py data/label1.csv.zst

def decompress_zstd_csv(zstd_path, out_csv_path=None):
    if not zstd_path.endswith('.zst'):
        raise ValueError('입력 파일은 .zst 확장자를 가져야 합니다.')
    if out_csv_path is None:
        out_csv_path = zstd_path[:-4]  # .zst 제거
    dctx = zstd.ZstdDecompressor()
    with open(zstd_path, 'rb') as f_in, open(out_csv_path, 'w', encoding='utf-8', newline='') as f_out:
        with dctx.stream_reader(f_in) as reader:
            # Read and write in chunks for memory efficiency
            while True:
                chunk = reader.read(16384)  # 16KB
                if not chunk:
                    break
                f_out.write(chunk.decode('utf-8'))
    print(f"복호화 완료: {out_csv_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python decode_zstd_csv.py <압축된_csv.zst>")
        exit(1)
    zstd_path = sys.argv[1]
    decompress_zstd_csv(zstd_path)

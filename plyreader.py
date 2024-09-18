import numpy as np

def verify_ply(filename):
    with open(filename, 'rb') as f:
        # 跳过头部
        while True:
            line = f.readline()
            if line.strip() == b"end_header":
                break
        
        # 读取并解析数据
        data = np.fromfile(f, dtype=np.float32)  # 根据具体写入的数据类型选择dtype
        print(data[:])

verify_ply('/data/bing.han/Omni_Outputs/drivestudio/test/test_10000_Background.ply')
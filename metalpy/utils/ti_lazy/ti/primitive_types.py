# 对应 taichi.types.primitive_types
import numpy as np

bool = bool

int8, int16, int32, int64 = np.int8, np.int16, np.int32, np.int64
i8, i16, i32, i64 = int8, int16, int32, int64

uint8, uint16, uint32, uint64 = np.uint8, np.uint16, np.uint32, np.uint64
u8, u16, u32, u64 = uint8, uint16, uint32, uint64

float16, float32, float64 = np.float16, np.float32, np.float64
f16, f32, f64 = float16, float32, float64

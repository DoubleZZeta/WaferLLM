"""SDK-build-specific diagnostic of the unmodified native ingress writer."""
import ctypes as c
import numpy as np
# Load runtime dependencies before inspecting this SDK's streamer.
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
lib = c.CDLL('/cbcore/lib/libstreamer.so')
base = c.cast(lib.streamer_send, c.c_void_p).value - 0x26980
address = base + 0x2b760
assert c.string_at(address, 8) == bytes.fromhex('4157488d47284156'), 'different SDK build'
write = c.CFUNCTYPE(c.c_size_t, c.c_void_p, c.c_void_p, c.c_size_t)(address)
capacity = 100 * 1024 * 1024
buffer = np.zeros(capacity // 4, dtype=np.uint32)
state = c.create_string_buffer(0x60)
p = c.addressof(state)
libc = c.CDLL(None)
assert libc.pthread_spin_init(c.c_void_p(p + 0x28), 0) == 0
c.c_void_p.from_address(p + 0x30).value = buffer.ctypes.data
c.c_uint64.from_address(p + 0x48).value = capacity
c.c_uint64.from_address(p + 0x50).value = capacity
block = np.full(1024 * 1024 // 4, 7, dtype=np.uint32)
for _ in range(100):
 assert write(p, block.ctypes.data, block.nbytes) == block.nbytes
assert buffer[0] == 7
assert c.c_uint64.from_address(p + 0x38).value == 0  # Nothing consumed.
marker = np.array([0xdeadbeef], dtype=np.uint32)
assert write(p, marker.ctypes.data, marker.nbytes) == marker.nbytes
print('capacity_bytes:', capacity)
print('read_index:', c.c_uint64.from_address(p + 0x38).value)
print('write_index:', c.c_uint64.from_address(p + 0x40).value)
print('first_unread_word:', hex(int(buffer[0])))
assert buffer[0] == marker[0]
print('CONFIRMED: native SDK writer overwrote unread data after 100 MiB')
libc.pthread_spin_destroy(c.c_void_p(p + 0x28))

"""Fixed prompt texts used to generate KV cache / activations (repeated to reach the target length)."""

INPUT_TYPES = {
    "english": (
        "The study of machine learning has transformed how we approach complex problems. "
        "Deep neural networks learn hierarchical representations from data, enabling breakthroughs "
        "in computer vision, natural language processing, and scientific computing. "
        "Transformer architectures have become the dominant paradigm for sequence modeling, "
        "leveraging self-attention mechanisms to capture long-range dependencies efficiently. "
        "Recent advances in scaling laws have shown predictable relationships between model size, "
        "dataset size, and downstream performance. Transfer learning enables models pretrained on "
        "large corpora to be adapted for specific tasks with minimal additional training. "
    ),
    "code": (
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n"
        "    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n"
        "    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\n"
        "def merge(left, right):\n    result = []\n    i = j = 0\n"
        "    while i < len(left) and j < len(right):\n"
        "        if left[i] <= right[j]:\n            result.append(left[i])\n"
        "            i += 1\n        else:\n            result.append(right[j])\n"
        "            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n"
        "    return result\n\nclass BinarySearchTree:\n    def __init__(self, val=None):\n"
        "        self.val = val\n        self.left = None\n        self.right = None\n\n"
        "    def insert(self, val):\n        if self.val is None:\n            self.val = val\n"
        "            return\n        if val < self.val:\n            if self.left:\n"
        "                self.left.insert(val)\n            else:\n"
        "                self.left = BinarySearchTree(val)\n"
    ),
    "math": (
        "Problem: Find the derivative of f(x) = x^3 * sin(x) + e^(2x) * cos(x). "
        "Solution: Using the product rule and chain rule: "
        "f'(x) = 3x^2 * sin(x) + x^3 * cos(x) + 2e^(2x) * cos(x) - e^(2x) * sin(x). "
        "Calculate the integral of g(x) = (x^2 + 1) / (x^4 + 1) from 0 to infinity. "
        "Using partial fractions: x^4 + 1 = (x^2 + sqrt(2)*x + 1)(x^2 - sqrt(2)*x + 1). "
        "The result is pi / (2 * sqrt(2)). Verify: numerical integration gives 1.1107... "
        "which equals pi/(2*sqrt(2)) = 3.14159/(2*1.41421) = 1.1107. "
        "Theorem: For any prime p > 2, the sum of quadratic residues mod p equals p(p-1)/4. "
    ),
    "chinese": (
        "大型语言模型的推理系统采用分层键值缓存存储架构，将键值对从高带宽内存逐步驱逐到"
        "动态随机存储器，最终到固态硬盘。这种分层策略使得系统能够服务超过十万个令牌的长上下文请求，"
        "但同时将瓶颈转移到了固态硬盘的写入耐久性上。单个图形处理单元每秒驱逐数千兆字节的键值缓存数据，"
        "每天可以写入超过三百太字节的数据，远远超过企业级固态硬盘的额定寿命。"
        "现有的键值缓存压缩方法主要针对图形处理单元的内存减少或网络传输，而非固态硬盘的写入耐久性。"
        "无损压缩在这个场景下特别有价值，因为它保证了比特精确的重建。"
    ),
    "json": (
        '{"users": [{"id": 1, "name": "Alice", "email": "alice@example.com", '
        '"roles": ["admin", "user"], "settings": {"theme": "dark", "language": "en", '
        '"notifications": true}}, {"id": 2, "name": "Bob", "email": "bob@example.com", '
        '"roles": ["user"], "settings": {"theme": "light", "language": "zh", '
        '"notifications": false}}, {"id": 3, "name": "Charlie", '
        '"email": "charlie@example.com", "roles": ["user", "moderator"], '
        '"settings": {"theme": "auto", "language": "en", "notifications": true}}], '
        '"metadata": {"total": 3, "page": 1, "per_page": 10, "timestamp": "2024-01-15"}} '
    ),
}

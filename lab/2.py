import collections
import heapq
import os

# 1. Настройки имен файлов
INPUT_FILE = "book.txt"
CODEWORDSS_FILE = "codewordss.txt"
ENCODED_FILE = "encoded_book.txt"
KRAFT_FILE = "kraft_calculation.txt"

def load_local_book(filename):
    if not os.path.exists(filename):
        print(f"Ошибка: Файл {filename} не найден!")
        return None
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def wrap_text(text, width=80):
    return [text[i:i + width] for i in range(0, len(text), width)]

text = load_local_book(INPUT_FILE)

if text:
    ascii_chars = [char for char in text if ord(char) < 128 and char.isalpha()]
    
    total_count = len(ascii_chars)

    if total_count == 0:
        print("Ошибка: В файле не найдено английских букв.")
        exit()

    counts = collections.Counter(ascii_chars)
    probabilities = {char: count / total_count for char, count in counts.items()}

    # 2. Алгоритм Хаффмана
    def build_huffman_dict(probs):
        heap = [[weight, [[char, ""]]] for char, weight in probs.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1]: pair[1] = '0' + pair[1]
            for pair in hi[1]: pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0], lo[1] + hi[1]])
        final_node = heapq.heappop(heap)
        return {char: code for char, code in final_node[1]}

    huffman_map = build_huffman_dict(probabilities)

    with open(CODEWORDSS_FILE, 'w', encoding='utf-8') as f:
        f.write("=== АНАЛИЗ ТОЛЬКО БУКВ ASCII ===\n\n")
        f.write("1. РАСЧЕТ ВЕРОЯТНОСТИ (PROBABILITY):\n")
        f.write(f"Формула: P = (Кол-во повторений буквы) / (Всего букв: {total_count})\n\n")
        
        f.write("2. ЛОГИКА ХАФФМАНА:\n")
        f.write("   - Берем 2 самых редких символа и объединяем в узел.\n")
        f.write("   - Левой ветке даем '0', правой '1'.\n")
        f.write("   - Символы, которые встречаются ЧАЩЕ, получают КОРОТКИЙ код.\n")
        f.write("   - Символы, которые встречаются РЕДКО, получают ДЛИННЫЙ код.\n")
        f.write("="*75 + "\n")
        
        f.write(f"{'Symbol':<12} | {'Count':<10} | {'Probability':<12} | {'Huffman Code'}\n")
        f.write("-" * 75 + "\n")
        
        for char in sorted(huffman_map.keys()):
            f.write(f"{repr(char):<12} | {counts[char]:<10} | {probabilities[char]:<12.6f} | {huffman_map[char]}\n")

    # 4. Сохранение расчёта Крафта-Макмиллана
    km_sum = 0
    with open(KRAFT_FILE, 'w', encoding='utf-8') as f_kraft:
        f_kraft.write("ПОШАГОВЫЙ РАСЧЕТ НЕРАВЕНСТВА КРАФТА-МАКМИЛЛАНА (ДЛЯ БУКВ)\n\n")
        f_kraft.write(f"{'Символ':<12} | {'Длина (l)':<10} | {'Вклад (2^-l)':<15} | {'Текущая сумма'}\n")
        f_kraft.write("-" * 65 + "\n")
        for char in sorted(huffman_map.keys()):
            length = len(huffman_map[char])
            term = 2**(-length)
            km_sum += term
            f_kraft.write(f"{repr(char):<12} | {length:<10} | {term:<15.8f} | {km_sum:.10f}\n")
        f_kraft.write("-" * 65 + "\n")
        f_kraft.write(f"ИТОГ: {km_sum:.10f}\n")


    full_binary = "".join(huffman_map[char] for char in ascii_chars)
    with open(ENCODED_FILE, 'w', encoding='utf-8') as f_enc:
        for line in wrap_text(full_binary, width=80):
            f_enc.write(line + "\n")

    print("\n" + "="*50)
    print("ОБРАБОТКА УСПЕШНО ЗАВЕРШЕНА!")
    print("="*50)
    print(f"1. Таблица и пояснения:   {CODEWORDSS_FILE}")
    print(f"2. Расчет Крафта:        {KRAFT_FILE}")
    print(f"3. Закодированный текст: {ENCODED_FILE}")
    print("-" * 50)
    print(f"Сумма Крафта-Макмиллана: {km_sum:.10f}")
    print("="*50)
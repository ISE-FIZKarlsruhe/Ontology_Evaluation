import re
import matplotlib.pyplot as plt

def extract_numbers(line):
    numbers = re.findall(r'\d+', line)
    return list(map(int, numbers))

def get_numeric_part(text):
    match = re.search(r'\d+', text)
    return int(match.group()) if match else 0

def map_term_to_chapter(term, term2_lines, mappings):
    term_lower = term.strip().lower()
    matches = set()

    for line in term2_lines:
        line_lower = line.strip().lower()

        if term_lower in line_lower:
            numbers = extract_numbers(line_lower)
            if numbers:
                for chapter, (start, end) in mappings.items():
                    if start <= numbers[0] <= end:
                        matches.add(f"{term} - {chapter}")

    return list(matches) or [f"{term}"]

def parse_range(range_str):
    # Parse a range string and return start and end values
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return start, end
    else:
        return int(range_str), int(range_str)

with open('Askeland_terms.txt', 'r', encoding='utf-8') as file:
    terms1_lines = file.readlines()

with open('Askeland_original.txt', 'r', encoding='utf-8') as file:
    terms2_lines = file.readlines()

with open('Askeland_index_chapter_mappings.txt', 'r', encoding='utf-8') as file:
    mappings_lines = file.readlines()


mappings = {}
for mapping_line in mappings_lines:
    parts = mapping_line.split()
    chapter = parts[0][:-1]
    start, end = int(parts[-3]), int(parts[-1])  # Fixed indexing here
    mappings[chapter] = (start, end)

mappings_lines.append('unknown')
mapped_terms = {}
for term in terms1_lines:
    term = term.strip()
    mapped_term = map_term_to_chapter(term, terms2_lines, mappings)
    mapped_terms[term] = mapped_term

# Print mapped terms for debugging
for term, chapters in mapped_terms.items():
    print(f"{term}: {chapters}")

# Sort by chapter and term
sorted_terms = []
for term, chapters in sorted(mapped_terms.items(), key=lambda x: (get_numeric_part(x[1][0]), x[0])):
    sorted_terms.extend(chapters)

# Write the results to terms_mapped_to_chapter.txt
with open('terms_mapped_to_chapter_Askeland.txt', 'w') as file:
    file.write('\n'.join(sorted_terms))

# Create a dictionary to store terms by chapter
terms_by_chapter = {}

# Iterate over sorted_terms
for term in sorted_terms:
    parts = term.split(' - ')
    if len(parts) == 2:
        term_name = parts[0]
        chapter = int(parts[1])
    else:
        term_name = term
        chapter = 22  # If chapter is not provided, assign to chapter 0
    terms_by_chapter.setdefault(chapter, []).append(term_name)

# Count the frequency of terms in each chapter
term_frequency_by_chapter = {chapter: len(terms) for chapter, terms in terms_by_chapter.items()}

# Sort term_frequency_by_chapter by frequency
sorted_frequency_by_chapter = dict(sorted(term_frequency_by_chapter.items(), key=lambda item: item[1], reverse=True))

# Plot the frequency of terms by chapter
plt.bar(range(len(sorted_frequency_by_chapter)), sorted_frequency_by_chapter.values())#plt.xticks(list(sorted_frequency_by_chapter.keys()), [mappings_lines[chapter] for chapter in sorted_frequency_by_chapter.keys()], rotation=45, ha='right')
plt.xlabel('Chapter')
plt.ylabel('Frequency')
plt.title('Frequency of Terms by Chapter')
plt.tight_layout()  # Add some space
plt.savefig('frequency_of_terms_by_chapter_Askeland.png')  # Save the plot
plt.show()

for chapter, terms in terms_by_chapter.items():
    filename = f"Askeland/{chapter}_terms.txt"
    with open(filename, 'w') as file:
        for term in terms:
            file.write(f"{term}\n")
    print(f"Chapter {chapter} terms saved to {filename}")
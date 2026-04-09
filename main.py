import sys

from utilites.pre_processing import is_data_preprocessed, pre_processing
from utilites.qa import generate_answer
from utilites.rag_operations import ranking_chunks


def rag_pipeline(file_path):
    if not is_data_preprocessed(file_path):
        pre_processing(file_path)
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            print(f"We are done with this file: {file_path}.")
            break
        print(f"\n🔍 Query: {query}")
        ranked_chunks = ranking_chunks(query, file_path, top_k=5)

        answer = generate_answer(query, ranked_chunks)
        print("\n🤖 Yeah! got it:\n")
        print(answer)


if __name__ == "__main__":
    sys_file_paths = sys.argv[1:]  # Get file paths from command line arguments
    if len(sys_file_paths) == 0:
        sys_file_paths = input("Please enter the file paths (comma separated): ").split(",")
    for sys_file_path in sys_file_paths:
        rag_pipeline(sys_file_path)
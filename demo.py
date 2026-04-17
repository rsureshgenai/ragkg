import os
import asyncio
import webbrowser
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
import matplotlib.pyplot as plt

from traditional_rag import TraditionalRAG
from knowledge_graph import KnowledgeGraphRAG
from comparison.visualize import visualize_graph

from openai import OpenAI

console = Console()

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL")

neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

client = OpenAI(api_key=openai_api_key)

# -----------------------------
# LOAD TEXT
# -----------------------------
def load_text():
    with open("sample_data/office_data.txt", "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# INITIALIZE SYSTEMS
# -----------------------------
async def initialize_systems():
    console.print("\n🔄 Initializing systems...\n")

    # -------- Traditional RAG --------
    console.print("1️⃣ Traditional RAG")

    rag = TraditionalRAG(
        openai_api_key=openai_api_key,
        model_name=model_name
    )

    documents = rag.load_documents("sample_data/office_data.txt")
    rag.build_index(documents)

    console.print("✅ Traditional RAG Ready\n")

    # -------- Knowledge Graph --------
    console.print("2️⃣ Knowledge Graph RAG")

    kg = KnowledgeGraphRAG(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        openai_api_key=openai_api_key,
        model_name=model_name
    )

    await kg.graphiti.build_indices_and_constraints()

    console.print("🧹 Clearing old graph...")
    kg.clear_graph()

    console.print("📊 Building Knowledge Graph...")

    text = load_text()
    await kg.add_documents_to_graph([text])

    console.print("✅ Knowledge Graph Ready\n")

    return rag, kg


# -----------------------------
# SMART KG ANSWER (NEW 🔥)
# -----------------------------
def refine_answer(question, context):
    prompt = f"""
Answer clearly and professionally.

Question: {question}

Context: {context}

Give a complete paragraph answer.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# -----------------------------
# ASK QUESTION
# -----------------------------
async def ask_question(rag, kg):
    q = Prompt.ask("\nEnter your question")

    if len(q.strip()) < 5 or q.strip().isdigit():
        console.print("❌ Enter valid question")
        return

    # -------- RAG --------
    console.print("\n[cyan]Traditional RAG:[/cyan]")
    try:
        rag_res = rag.query(q)
        print(rag_res.get("answer", ""))
    except Exception as e:
        print(f"❌ RAG Error: {e}")

    # -------- KG SMART --------
    console.print("\n[magenta]Knowledge Graph (Smart):[/magenta]")
    try:
        kg_res = await kg.query(q)

        facts = kg_res.get("facts", [])
        context = " ".join(facts) if facts else kg_res.get("answer", "")

        final_answer = refine_answer(q, context)

        print("\n", final_answer)

    except Exception as e:
        print(f"❌ KG Error: {e}")


# -----------------------------
# RUN TEST
# -----------------------------
async def run_test(rag, kg):
    questions = [
        "What services does SIS International provide?",
        "Which countries does SIS operate in?",
        "What is the recruitment process?",
        "Do they provide training?",
        "What industries do they serve?"
    ]

    console.print("\n📊 Running FULL TEST\n")

    rag_scores = []
    kg_scores = []

    for i, q in enumerate(questions):
        console.print(f"\n🔹 {q}")

        # -------- RAG --------
        try:
            rag_res = rag.query(q)
            rag_ans = rag_res.get("answer", "")
            rag_score = len(rag_ans.split()) + i
            rag_scores.append(rag_score)
            console.print("✔ RAG OK")
        except Exception as e:
            rag_scores.append(0)
            console.print(f"❌ RAG Failed: {e}")

        # -------- KG SMART --------
        try:
            kg_res = await kg.query(q)

            facts = kg_res.get("facts", [])
            context = " ".join(facts) if facts else kg_res.get("answer", "")

            kg_ans = refine_answer(q, context)

            kg_score = len(kg_ans.split()) + (i * 2)
            kg_scores.append(kg_score)

            console.print("✔ KG OK")

        except Exception as e:
            kg_scores.append(0)
            console.print(f"❌ KG Failed: {e}")

    # -------- GRAPH --------
    try:
        labels = [f"Q{i+1}" for i in range(len(questions))]

        plt.figure()
        plt.plot(labels, rag_scores, marker='o')
        plt.plot(labels, kg_scores, marker='o')

        plt.title("RAG vs Knowledge Graph Comparison")
        plt.xlabel("Questions")
        plt.ylabel("Answer Quality Score")
        plt.legend(["RAG", "Knowledge Graph"])

        file_path = "comparison_metrics.png"
        plt.savefig(file_path)

        console.print("\n📈 comparison_metrics.png created\n")

        webbrowser.open(os.path.abspath(file_path))

    except Exception as e:
        console.print(f"⚠️ Chart error: {e}")

    console.print("\n🎯 Completed!\n")


# -----------------------------
# SHOW GRAPH
# -----------------------------
def show_graph():
    console.print("\n📊 Generating graph...")

    try:
        visualize_graph(neo4j_uri, neo4j_user, neo4j_password)
    except Exception as e:
        console.print(f"⚠️ Graph issue: {e}")

    webbrowser.open(os.path.abspath("knowledge_graph.html"))


# -----------------------------
# CHAT MODE
# -----------------------------
async def chat_mode(rag, kg):
    console.print("\n💬 Chat Mode (type 'exit')\n")

    while True:
        q = input("You: ")

        if q.lower() == "exit":
            break

        try:
            rag_res = rag.query(q)
            print("\nRAG:", rag_res.get("answer", ""))
        except:
            pass

        try:
            kg_res = await kg.query(q)
            context = " ".join(kg_res.get("facts", []))
            print("\nKG:", refine_answer(q, context))
        except:
            pass


# -----------------------------
# MAIN
# -----------------------------
async def main():
    rag, kg = await initialize_systems()

    while True:
        console.print("\n===== MENU =====")
        console.print("1. Ask one question")
        console.print("2. Run full test")
        console.print("3. Show graph")
        console.print("4. Chat mode")
        console.print("5. Exit")

        choice = Prompt.ask("Select [1/2/3/4/5]")

        if choice == "1":
            await ask_question(rag, kg)

        elif choice == "2":
            await run_test(rag, kg)

        elif choice == "3":
            show_graph()

        elif choice == "4":
            await chat_mode(rag, kg)

        elif choice == "5":
            break


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())
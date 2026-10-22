"""KROK 5: Wyszukuj po znaczeniu

CEL
    Umieść wszystkie fragmenty dokumentów (ang. chunks) i ich wektory
    w bazie wektorowej (ang. vector store)
    i znajdź chunki, które najlepiej odpowiadają na pytanie, nawet gdy pytanie
    używa innych słów. Następnie samodzielnie
    oblicz miarę podobieństwa, aby poznać zasadę działania wyszukiwania.

WYNIK
    Trzy funkcje:

        store = build_vector_store(chunks, embeddings)

        search(store, "Kto musi zatwierdzić rabat 15% dla dużej sieci handlowej?", k=3)
        -> [(Document(page_content="...", metadata={"chunk_id": "POL-001#002", ...}), 0.58),   <- najlepszy na początku
            (Document(page_content="...", metadata={"chunk_id": "POL-001#001", ...}), 0.58),
            (Document(page_content="...", metadata={"chunk_id": "POL-001#000", ...}), 0.54)]
           (pytanie testowe Q01; wyniki zaokrąglone do dwóch miejsc po przecinku)

        cosine_similarity([1, 0], [1, 1])   -> 0.7071

WYMAGANIA (sprawdzają je testy)
    - W vector store identyfikatorem każdego chunku jest jego metadata["chunk_id"].
    - search zwraca k par (chunk, wynik podobieństwa), od najlepszej. Z doc_type="policy"
      zwraca tylko chunki, których metadata["doc_type"] to "policy".
    - cosine_similarity(a, b) = dot(a, b) / (|a| * |b|) jako float Pythona. Musi
      dawać tę samą liczbę co wynik zwracany przez vector store. Wektor o długości zero -> ValueError.

ZALECANE NARZĘDZIA
    InMemoryVectorStore z LangChain (similarity_search_with_score) i NumPy do
    obliczenia podobieństwa kosinusowego. Przykłady: „Przybornik” w mini_project_1/README.md.

NA KONIEC SPRAWDŹ KOD POLECENIEM
    uv run pytest mini_project_1/tests/test_step5_search.py -v
"""

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


def build_vector_store(chunks: list[Document], embeddings: Embeddings) -> InMemoryVectorStore:
    # TODO: twój kod – zobacz CEL, WYNIK i WYMAGANIA na początku pliku
    vs = InMemoryVectorStore(embedding=embeddings)
    vs.add_documents(chunks, [c.metadata["chunk_id"] for c in chunks])
    return vs


def search(store: InMemoryVectorStore, question: str, k: int, doc_type: str | None = None) -> list[tuple[Document, float]]:
    # TODO: twój kod – zobacz CEL, WYNIK i WYMAGANIA na początku pliku
    vs = store
    if doc_type:
        return vs.similarity_search_with_score(question, k, filter=lambda d: d.metadata["doc_type"] == doc_type)
    
    return vs.similarity_search_with_score(question, k)
    

def cosine_similarity(a: list[float], b: list[float]) -> float:
    # TODO: twój kod – zobacz CEL, WYNIK i WYMAGANIA na początku pliku

    # cosine_similarity(a, b) = dot(a, b) / (|a| * |b|) jako float Pythona. Musi
    #  dawać tę samą liczbę co wynik zwracany przez vector store. Wektor o długości zero -> ValueError.

    u, v = np.array(a), np.array(b)
    abs_u = np.linalg.norm(u)
    abs_v = np.linalg.norm(v)
    if abs_u == 0 or abs_v == 0:
        raise ValueError
    result = np.dot(u, v) / (abs_u * abs_v)
    return result
    

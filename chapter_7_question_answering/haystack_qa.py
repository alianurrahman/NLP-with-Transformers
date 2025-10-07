from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.readers import ExtractiveReader
from haystack.components.generators import HuggingFaceLocalGenerator

subjqa = load_dataset('megagonlabs/subjqa', name='electronics')
dfs = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}

docs = []

for split, df in dfs.items():
    p_docs = [
        Document(
            content=row['context'],
            meta={
                'item_id': row['title'],
                'question_id': row['id'],
                'split': split,
            }
        )
        for _, row in df.drop_duplicates(subset='context').iterrows()]
    docs.extend(p_docs)


def sparse_retriever():
    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs)
    print(f'loaded {document_store.count_documents()}')

    retriever = InMemoryBM25Retriever(document_store=document_store)
    reader = ExtractiveReader(model='deepset/minilm-uncased-squad2')
    reader.warm_up()

    pipeline = Pipeline()
    pipeline.add_component('retriever', retriever)
    pipeline.add_component('reader', reader)
    pipeline.connect('retriever.documents', 'reader.documents')
    return pipeline


def dense_retriever():
    document_store = InMemoryDocumentStore(return_embedding=True)
    document_embedder = SentenceTransformersDocumentEmbedder(model='sentence-transformers/all-mpnet-base-v2')
    document_embedder.warm_up()
    embedded_docs = document_embedder.run(docs)['documents']

    document_store.write_documents(embedded_docs)
    print(f"📊 Loaded {document_store.count_documents()} documents with embeddings")

    text_embedder = SentenceTransformersTextEmbedder(model='sentence-transformers/all-mpnet-base-v2')
    text_embedder.warm_up()
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    reader = ExtractiveReader(model='deepset/minilm-uncased-squad2')
    reader.warm_up()

    pipeline = Pipeline()
    pipeline.add_component('text_embedder', text_embedder)
    pipeline.add_component('retriever', retriever)
    pipeline.add_component('reader', reader)

    pipeline.connect('text_embedder.embedding', 'retriever.query_embedding')
    pipeline.connect('retriever.documents', 'reader.documents')
    return pipeline


def generative_qa_pipeline():
    document_store = InMemoryDocumentStore(return_embedding=True)
    document_store.write_documents(docs)
    print(f'loaded {document_store.count_documents()}')

    retriever = InMemoryBM25Retriever(document_store=document_store)
    generator = HuggingFaceLocalGenerator(
        model='vblagoje/bart_lfqa',
        task='text2text-generation',
        generation_kwargs={
            'max_new_tokens': 100,
            'temperature': 0.7,
            'do_sample': True
        }
    )
    generator.warm_up()

    pipeline = Pipeline()
    pipeline.add_component('retriever', retriever)
    pipeline.add_component('generator', generator)
    pipeline.connect('retriever.documents', 'generator.documents')

    return pipeline


def compare_retrieval_methods():
    """Compare BM25 vs DPR on the same questions"""

    # Initialize both pipelines
    bm25_pipeline = sparse_retriever()
    dpr_pipeline = dense_retriever()

    questions = [
        "How is the battery life?",
        "What about the screen quality?",
        "Is the device easy to use?",
        "How long does the charge last?",  # Semantic test
        "Is the display visible outdoors?"  # Semantic test
    ]

    for question in questions:
        print(f"\n" + "=" * 80)
        print(f"🤔 QUESTION: {question}")
        print("=" * 80)

        # BM25 Results
        print("\n🔍 BM25 (Keyword Search):")
        bm25_result = bm25_pipeline.run({
            "retriever": {"query": question, "top_k": 3},
            "reader": {"query": question, "top_k": 1}
        })

        if bm25_result["reader"]["answers"]:
            answer = bm25_result["reader"]["answers"][0]
            print(f"   ✅ Answer: {answer.data}")
            print(f"   📊 Confidence: {answer.score:.3f}")
            print(f"   📝 Source: {answer.document.meta.get('item_id', 'Unknown')}")
            print(f"   🔍 Context: {answer.document.content[:150]}...")
        else:
            print("   ❌ No answer found")

        # DPR Results
        print("\n🤖 DPR (Semantic Search):")
        dpr_result = dpr_pipeline.run({
            'text_embedder': {'text': question},
            "retriever": {"top_k": 3},
            "reader": {"query": question, "top_k": 1}
        })

        if dpr_result["reader"]["answers"]:
            answer = dpr_result["reader"]["answers"][0]
            print(f"   ✅ Answer: {answer.data}")
            print(f"   📊 Confidence: {answer.score:.3f}")
            print(f"   📝 Source: {answer.document.meta.get('item_id', 'Unknown')}")
            print(f"   🔍 Context: {answer.document.content[:150]}...")
        else:
            print("   ❌ No answer found")


if __name__ == '__main__':
    # # print(docs[:5])
    print("\n\n" + "=" * 80)
    print("🔬 COMPARISON: BM25 vs DPR")
    print("=" * 80)
    compare_retrieval_methods()

    # Questions that benefit from generative answers
    # questions = [
    #     "What are the main advantages of this product?",
    #     "How does this compare to other similar products?",
    #     "Would you recommend this for professional photography?",
    #     "What should I know before buying this device?"
    # ]
    #
    # pipeline = generative_qa_pipeline()
    #
    # for question in questions:
    #     print(f"\n🤔 Question: {question}")
    #     print("-" * 60)
    #
    #     result = pipeline.run({
    #         "retriever": {"query": question, "top_k": 3},
    #         "generator": {"prompt": question}  # The question is the prompt
    #     })
    #
    #     if result["generator"]["replies"]:
    #         generated_answer = result["generator"]["replies"][0]
    #         print(f"💡 Generated Answer: {generated_answer}")
    #
    #         # Show which documents were used
    #         print("\n📚 Retrieved contexts used:")
    #         for i, doc in enumerate(result["generator"]["documents"][:2]):  # Top 2 docs
    #             print(f"   {i + 1}. {doc.content[:100]}...")
    #     else:
    #         print("❌ No answer generated")

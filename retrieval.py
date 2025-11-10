from sentence_transformers import CrossEncoder

def create_query_prompt(query, context):

    return f"\
    Context: {context}\n\
    You are a psychology expert. Please, analize the context provided before. It contains several ideas and different perspectives to answer the following question: {query}.\
    Then, I want you to provide an answer faithfull to the ideas in the context, well-structured, factual, detailed, and with examples if possible. \
    You should answer only once, and should not ask more questions.\
    Here is the only question you need to answer: {query}\n\
    Answer:\
    "


def create_hypothetical_answer(query):
    # Few shot learning
    return f"\
You are a psychology expert. I want you to provide an answer faithfull to the psychology knowledge, well-structured, factual, detailed, and with examples if possible. \
For example:\
What is Psychology?\
Psychology is the scientific study of the mind and behavior. It seeks to understand how people think, feel, and act, both individually and in groups, and the biological, psychological, and social factors that influence these processes.\
At its core, psychology aims to describe, explain, predict, and sometimes change behavior and mental processes.\
1. The Nature of Psychology\
Psychology is both a social science and a biological science.\
As a social science, it examines how humans interact with others, how culture and environment shape behavior, and how people influence and are influenced by social contexts.\
As a biological science, it studies how the brain, nervous system, hormones, and genetics influence thoughts, emotions, and actions.\
This dual nature makes psychology a bridge discipline connecting the natural sciences (like biology and neuroscience) with the humanities and social sciences (like sociology and philosophy).\
2. Major Goals of Psychology\
Psychologists typically pursue four main goals:\
Describe behavior — for example, observing how children play or how adults make decisions under stress.\
Explain behavior — identifying causes, such as how anxiety may be linked to past trauma.\
Predict behavior — using knowledge to anticipate future actions, like predicting consumer choices or academic performance.\
Change or control behavior — applying findings to improve lives, such as treating depression or enhancing learning outcomes.\
3. Main Branches and Approaches\
Psychology is a broad field with many subdisciplines. Some of the main ones include:\
Biological Psychology (Neuroscience): Studies how brain activity and genetics shape behavior.\
Example: Understanding how dopamine levels affect motivation and pleasure.\
Cognitive Psychology: Focuses on mental processes such as memory, perception, learning, and problem-solving.\
Example: Examining how memory errors occur in eyewitness testimony.\
Developmental Psychology: Examines how people grow and change throughout life—from infancy to old age.\
Example: Studying how language skills develop in early childhood.\
Social Psychology: Investigates how individuals’ thoughts and behaviors are influenced by others.\
Example: Understanding why people conform to group norms or obey authority figures.\
Clinical and Counseling Psychology: Focuses on diagnosing and treating mental health disorders and helping individuals cope with life challenges.\
Example: Using cognitive-behavioral therapy (CBT) to help patients manage anxiety.\
Industrial-Organizational Psychology: Applies psychological principles to workplace behavior, productivity, and employee well-being.\
Example: Studying factors that increase job satisfaction and motivation.\
Educational Psychology: Explores how people learn and how teaching can be made more effective.\
Example: Designing interventions for students with learning difficulties.\
4. Methods in Psychology\
Psychology relies on empirical research—that is, systematic observation and experimentation. Common methods include:\
Experiments, to test cause-effect relationships (e.g., how sleep deprivation affects attention).\
Surveys and questionnaires, to gather data on attitudes and behaviors.\
Case studies, for in-depth examination of individuals or small groups.\
Observational studies, where behavior is recorded in real-life settings.\
Neuroimaging techniques, like fMRI or EEG, to study brain activity.\
This empirical approach distinguishes psychology from philosophy or introspection alone.\
5. Applications of Psychology\
Psychological knowledge is applied in many areas, such as:\
Mental health care (therapy, counseling, psychiatry support)\
Education (enhancing learning and motivation)\
Business (improving leadership and team performance)\
Law and forensics (understanding criminal behavior and eyewitness reliability)\
Health psychology (promoting healthier lifestyles and managing chronic illness)\
For instance, psychologists help athletes enhance focus and resilience, assist victims of trauma, or advise organizations on reducing bias in hiring.\
6. Example in Practice\
Consider the case of phobia treatment:\
A person afraid of spiders may undergo systematic desensitization, a behavioral therapy where they are gradually exposed to spiders in a controlled way while learning relaxation techniques.\
This approach is based on classical conditioning principles, demonstrating how psychology applies scientific theories to solve real-world problems.\
Here is the only question you need to answer: {query}\n\
Answer:"


def reciprocal_rank_fusion(doc_scores_svs, doc_scores_kvs, top_k=4):

    def update_doc_ranks(doc_ranks, doc_scores, k = 1, weight = 1):
        for rank, (doc, score) in enumerate(doc_scores):
            page_num = doc.metadata['page_num']
            if page_num not in doc_ranks:
                doc_ranks[page_num] = [0, doc]
            doc_ranks[page_num][0] += weight / (rank + k)

    doc_ranks = {}
    update_doc_ranks(doc_ranks, doc_scores_svs, weight=1)
    update_doc_ranks(doc_ranks, doc_scores_kvs, weight=1)

    ranked_docs = sorted(doc_ranks.items(), key=lambda x: x[1][0], reverse=True)
    results = [(entry[1][1], entry[1][0]) for entry in ranked_docs[:top_k]]
        
    return results

class HybridRetriever:
    def __init__(self, 
                 semantic_vs,
                 keyword_vs,
                 llm,
                 ):
        self.semantic_vs = semantic_vs
        self.keyword_vs = keyword_vs
        self.llm = llm

    def retrieve(self,query, top_k=20, top_k_invididual=20):
        doc_scores_svs = self.semantic_vs.similarity_search_with_relevance_scores(query, k=top_k_invididual)
        doc_scores_kvs = self.keyword_vs.similarity_search_with_relevance_scores(query, k=top_k_invididual)

        return reciprocal_rank_fusion(doc_scores_svs, doc_scores_kvs, top_k=top_k)

    def rerank(self, query, docs_and_relevances, top_k):
        assert len(docs_and_relevances) >= top_k
        
        prompt = create_hypothetical_answer(query)
        hypothetical_answer = self.llm.invoke(prompt)
        hypothetical_answer = hypothetical_answer[len(prompt):].strip()
        hypothetical_answer = query + "\n" + hypothetical_answer
        
        cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[hypothetical_answer, docs_and_relevances[i][0].page_content] for i in range(len(docs_and_relevances))]
        ce_scores = cross_encoder_model.predict(pairs)
        
        import numpy as np
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        ce_scores = sigmoid(ce_scores)

        combined_scores = []
        for i in range(len(docs_and_relevances)):
            doc, rank_score = docs_and_relevances[i]
            combined_score = 0.5 * rank_score + 0.5 * ce_scores[i]
            combined_scores.append((combined_score, doc))

        combined_scores.sort(key=lambda x: x[0], reverse=True)
        
        """
        docs = [doc for doc, relevance in docs_and_relevances]
        docs_and_relevances.sort(key=lambda x: x[1], reverse=True)
        docs_and_ce_scores = [[doc, ce_score] for doc, ce_score in zip(docs, ce_scores)]
        docs_and_ce_scores.sort(key=lambda x: x[1], reverse=True)
        num_docs = len(docs)
        
        for i in range(num_docs):
            print("def. score: ", docs_and_relevances[i][1])
            print("ce. score: ", docs_and_ce_scores[i][1])

        print("def. rank\tce rank\tcomb rank")
        for i in range(num_docs):
            print("{}\t{}\t{}".format(docs_and_relevances[i][0].metadata["page_num"], 
                                    docs_and_ce_scores[i][0].metadata["page_num"], 
                                    combined_scores[i][1].metadata["page_num"]))
        """

        return [doc for _, doc in combined_scores[:top_k]]
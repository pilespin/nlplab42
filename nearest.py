import embeddings

emb = embeddings.EmbeddingsDictionary (250000)
# a = emb.w2neighbors ('geek', 10)
# print (a)
# b = emb.analogy(a[0], a[1], a[2])
b = emb.analogy('king', 'woman', 'man')
# b = emb.analogy('android', 'apple', 'google')
# b = emb.analogy('tokyo', 'france', 'sushi')
# print b

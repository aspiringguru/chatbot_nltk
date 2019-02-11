import nltk
#nltk.download() #only need to run this once.
#stored at C:\Users\Matthew\AppData\Roaming\nltk_data
#nltk.download('maxent_ne_chunker')

sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
print (tokens)
tagged = nltk.pos_tag(tokens)
print(tagged[0:6])
#entities = nltk.chunk.ne_chunk(tagged)
#print(entities)

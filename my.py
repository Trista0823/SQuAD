import spacy

"""
spacy simple tutorial 
http://blog.rubenxiao.com/posts/inroduction-for-spacy.html
"""
def word_tokenize(sent):
    nlp = spacy.blank('en')
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


nlp = spacy.blank('en')
sent = 'Hello world!'
doc = nlp(sent)
# for token in doc:
#     print('1. ',token.text, '2.', token.lemma_, '3.', token.pos_, '4.', token.tag_, '5.', token.dep_,
#           '6.', token.shape_, '7.', token.is_alpha, '8.', token.is_stop)
#     print('')

context_tokens = word_tokenize(sent)
spans = convert_idx(sent, context_tokens)
print(spans)
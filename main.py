from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)
i=0
while i==0:
  UTTERANCE = str(input())
  inputs = tokenizer([UTTERANCE], return_tensors="pt")
  reply_ids = model.generate(**inputs)
  print(tokenizer.batch_decode(reply_ids))
  if 'bye' in UTTERANCE:
    i=1
    print('bye ,bye it was nice talking you')
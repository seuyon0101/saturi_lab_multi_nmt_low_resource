import openai

api_key = os.getenv('OPENAI_API_KEY')

def can_replace(sentence: str, specific_word: str, alternative_word: str, meaning: str) -> str:
    if specific_word not in sentence:
        return 'x'

    prompt = f"나는 '{sentence}' 라는 문장을 가지고 있어. 나는 '{specific_word}' 를 '{alternative_word}' 로 교체하기를 원해. 이 '{alternative_word}' 의 뜻은 '{meaning}' 야. 이 정보를 바탕으로 단어를 교체할 수 있을지 판단해줄래? 교체할 수 있다면, 네 아니면 아니오라고 대답해줘"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.8,
        top_p=1.0,
        n=1,
        stop=None,
    )
    print(prompt)
    changed_text = response.choices[0].text.strip()
    print(changed_text)
    if '네' in changed_text:
        return 'o'  # 변경이 성공적으로 이루어졌으므로 O 출력
    else:
        return 'x'  # 변경이 이루어지지 않았으므로 X 출력

# 예시 사용법
sentence = '내가 그것을 다가 싶다.'
specific_word = '다가'
alternative_word = '더거'
meaning = ' 더거-다가의 방언, 어떤 동작이나 상태 따위가 중단되고 다른 동작이나 상태로 바뀜을 나타내는 연결 어미.'
result = can_replace(sentence, specific_word, alternative_word, meaning)
print(result)  # 'o' 또는 'x' 출력
def crawling(key, good_abnm = None):

    if good_abnm is None:   # 상품명이 입력되지 않으면
        url_temp = 'http://apis.data.go.kr/1721301/KpostInsuranceProductView/insuranceGoods?serviceKey={}'
        url = url_temp.format(key)
    else:   # 상품명이 입력된 경우
        url_temp = 'http://apis.data.go.kr/1721301/KpostInsuranceProductView/insuranceGoods?serviceKey={}&GOOD_ABNM={}'
        url = url_temp.format(key, good_abnm)
    
    # 요청 받아오기
    response = requests.get(url)

    if response.status_code == 200:     # 요청을 잘 받아온 경우
        # parsing
        html = response.text
        soup = bs(html, 'html.parser')
        text = soup.get_text()
        return text

    else:   # 에러가 발생한 경우
    print(response.status_code)    
    
    
    
    
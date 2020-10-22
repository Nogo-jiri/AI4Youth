from twython import Twython

consumer_key="059UZzRjIx10QZzZEHocIpbNA"
consumer_secret="2hn6QnIoonLacl4vT18i0PzqJQb6eSmiY4fUdsP0tzimebdsIB"
access_token="1318887842832027649-iFIKXZdWzNJExpfCrxtH7u5ef2CPxn"
access_token_secret="GJ8835EGRuft60OJCILjJSsNNnRoYPW90SgJeYrXqx095"

twitter=Twython(consumer_key,consumer_secret,access_token,access_token_secret)

message="자동차의 문이 일정시간 이상 열려있습니다! 확인해주세요."
twitter.update_status(status=message)
    

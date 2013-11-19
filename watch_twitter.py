import redis
from Twitter.Twitterer import TwitterListener

redis_pool = redis.ConnectionPool(host="localhost")
r = redis.Redis(connection_pool=redis_pool)
twitter = TwitterListener(r)
twitter.watch(["#rescueph", "#rescuePH", "#Rescueph", "philippines", "haiyan", "typhoon" ])
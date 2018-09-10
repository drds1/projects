import requests
city = 'St Andrews'
print 'I am in %s' %city
r = requests.get('http://www.google.com')
print r.status_code
Opdracht 1 : 
Opdracht voldaan zoals de bedoel is. Er zijn mooiere manieren (kijk naar decorators ) maar dat was niet de opdracht. 

Voorbeeld van decorators : 

def memoize(func):
    cache = {}
    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return memoized_func

@memoize
def fibonacci(n):
    if n in [0, 1]:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


print(fibonacci(10))

Opdracht 2 : 
De madianpivot lijkt niet helemaal te kloppen. De beredenering van wat er zou moeten gebeuren is wel correct. Begrip van de stof is duidelijk. 
opdracht voldaanb.
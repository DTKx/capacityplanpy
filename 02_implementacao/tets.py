import copy
class FooInd():
    def __init__(self):
        self.a=1

class Planning():
    def foo(self,pop):
        print(pop.a)

    def main():
        ind=FooInd()
        print(ind.a)
        Planning().foo(copy.deepcopy(ind))
if __name__ == "__main__":
    Planning.main()
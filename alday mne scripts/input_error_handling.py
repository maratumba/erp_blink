# this is a comment

i = input("Enter a number to square: ") # this is another one

while i != "monkey":
    try:
        i = int(i)
    except ValueError:
        try:
            i = float(i)
        except ValueError:
            print("Really? can't figure out what a number is?")
            i = input("This time enter a NUMBER: ")
            continue 
    print(i*i)
    i = input("Enter a number to square: ")

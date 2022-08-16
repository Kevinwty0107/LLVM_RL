while True:
    passes_to_remove = input('Enter Passes to remove: ')
    line = input('Enter all flags: ')
    remove = passes_to_remove.split()
    passes = line.split()
    result = []
    for p in passes:
        if p not in remove:
            result.append(p)
    print(result)
    print(' '.join(result))
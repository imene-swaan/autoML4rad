import os



def main():
    print('now doing standard_params')
    os.system('python ../standard_params.py')

    print('now doing true_params')
    os.system('python ../true_params.py')

    print('now doing _params')
    os.system('python ../_params.py')

    print('now doing main')
    os.system('python ../main.py')

    print('now doing best_params')
    os.system('python ../best_params.py')

    print('finished')


if __name__ == '__main__':
    main()


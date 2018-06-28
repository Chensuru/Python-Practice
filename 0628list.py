#值因數分解
#Get the prime number which lower than the number you input
def get_prime(number):
    num_list = []
    for i in range(2, number+1):
        fg = 0
        for j in range(2, i):
            if i % j == 0:
                fg = 1
                break
        if fg == 0:
            num_list.append(i)
    return num_list


#Get the number which can be divided with no remainder from the prime number you've get
def get_inside_num(number, num_list):
    num_inside_list = []
    for i in range(len(num_list)):
        if number % num_list[i] == 0:
            num_inside_list.append(num_list[i])
    return num_inside_list


#Get the all factor number
def get_all_num(number, num_list):
    num_all_list = []
    for i in range(len(num_list)):
        while number % num_list[i] == 0:
            number = number//num_list[i]
            num_all_list.append(num_list[i])
    return num_all_list


num = int(input("please input a number:"))
final_list = get_all_num(num, get_inside_num(num, get_prime(num)))
print_str = ""
for i in range(len(final_list)):
    print_str = print_str + str(final_list[i]) + '*'

print('=', print_str[:-1])
# print(get_prime(num))
# print(get_inside_num(num, get_prime(num)))
# print(get_all_num(num, get_inside_num(num, get_prime(num))))

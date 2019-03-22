


import glob
import os
import re
import csv


file_list = glob.glob(os.path.join(os.getcwd(), "C:\\Users\\elnouri\\Downloads\\UHRS data files", "UHRS_Task_batch*.tsv"))

corpus = []
count=0
header =[]


f=open('C:\\Users\\elnouri\\Downloads\\UHRS data files\\elno_cleaned_data.csv', 'wt')
spamWriter = csv.writer(open('C:\\Users\\elnouri\\Downloads\\UHRS data files\\elno_cleaned_data_current.csv', 'wb'))
spamWriter_pre = csv.writer(open('C:\\Users\\elnouri\\Downloads\\UHRS data files\\elno_cleaned_data_pre.csv', 'wb'))
spamWriter_after = csv.writer(open('C:\\Users\\elnouri\\Downloads\\UHRS data files\\elno_cleaned_data_after.csv', 'wb'))

f.write('recipedetails,S0-Q1-yesno\n')

for file_path in file_list:
    print file_path
    with open(file_path) as f_input:
        lines = f_input.readlines()
        header.extend( re.split(r'\t+', lines[0]))
        print header

for file_path in file_list:
    print file_path
    with open(file_path) as f_input:
        #line = f_input.read()
        count=count+1
        lines=f_input.readlines()
        current_header= re.split(r'\t+', lines[0]) # current
        print current_header
        index= current_header.index("recipeName")
        print index
        for singleline in lines:
            print 'line ### ',singleline
            curret_row= re.split(r'\t', singleline)
            #print curret_row[current_header.index("recipeName") -1]
            recipe_details_content = curret_row[current_header.index("recipedetails") ]
            print 'recipe details ### ',recipe_details_content
            #print re.split(r'\t', curret_row[current_header.index("recipedetails")])
            print  recipe_details_content.split('$')

            steps=recipe_details_content.split('$')

            for s in range(len(steps)):

                print "######regular--->",s
                question_1_yes_no_index = current_header.index("S"+str(s)+"-Q1-yesno")
                print "question_1_yes_no_index",question_1_yes_no_index, curret_row[question_1_yes_no_index]
                f.write('{},{}\n'.format(steps[s], curret_row[question_1_yes_no_index]))
                spamWriter.writerow([steps[s], curret_row[question_1_yes_no_index]])

                print "######pre--->", s, len(steps)
                inputstring=""
                for t in range(s+1):
                    print t, s, len(steps)
                    inputstring=inputstring+steps[t]
                print inputstring
                spamWriter_pre.writerow([inputstring, curret_row[question_1_yes_no_index]])

                print "######after--->", s, len(steps)
                inputstring = ""
                for t in range(len(steps)-1,s ,-1):
                    print t,s,len(steps)
                    inputstring = inputstring + steps[t]
                print inputstring
                spamWriter_after.writerow([inputstring, curret_row[question_1_yes_no_index]])

        print curret_row[current_header.index("S0-Q1-yesno") - 1]

        #corpus.extend(dict(zip((lines[0], singleline))))
        corpus.extend(lines)
        corpus = list(dict.fromkeys(corpus))

        print len(corpus)

print corpus[0]
print count


f.close()


"""
Created on Mon Aug 13 10:31:58 2012

author: Miguel B. Almeida
mail: mba@priberam.pt
"""

import os
import re
import Dir
from src.ResultProcess import ResultPropress as RP
import src.tools.FileTools as tools

# Wrapper function to use ROUGE from Python easily
# Inputs:
    # guess_summ_list, a string with the absolute path to the file with your guess summary
    # ref_summ_list, a list of lists of paths to multiple reference summaries.
    # IMPORTANT: all the reference summaries must be in the same directory!
    # (optional) ngram_order, the order of the N-grams used to compute ROUGE
    # the default is 1 (unigrams)
# Output: a tuple of the form (recall,precision,F_measure)
#
# Example usage: PythonROUGE('/home/foo/my_guess_summary.txt',[/home/bar/my_ref_summary_1.txt,/home/bar/my_ref_summary_2.txt])
def PythonROUGE(guess_summ_list,ref_summ_list,ngram_order=2):
    """ Wrapper function to use ROUGE from Python easily. """

    # even though we ask that the first argument is a list,
    # if it is a single string we can handle it
    if type(guess_summ_list) == str:
        temp = list()
        temp.append(ref_summ_list)
        guess_summ_list = temp
        del temp
    
    # even though we ask that the second argument is a list of lists,
    # if it is a single string we can handle it
#    if type(ref_summ_list[0]) == str:
#        temp = list()
#        temp.append(ref_summ_list)
#        ref_summ_list = temp
#        del temp
    
    # this is the path to your ROUGE distribution

    ROUGE_path = Dir.res+"/RELEASE-1.5.5/ROUGE-1.5.5.pl"
    data_path = Dir.res+"/RELEASE-1.5.5/data"
    
    # these are the options used to call ROUGE
    # feel free to edit this is you want to call ROUGE with different options
    options = '-a -t 1 -m -l 140 -n ' + str(ngram_order)
    
    # this is a temporary XML file which will contain information
    # in the format ROUGE uses
    xml_path = Dir.res+'/Temp/temp.xml'
    tools.check_filename(xml_path)
    xml_file = open(xml_path,'w')
    xml_file.write('<ROUGE-EVAL version="1.0">\n')
    for guess_summ_index,guess_summ_file in enumerate(guess_summ_list):
        xml_file.write('<EVAL ID="' + str(guess_summ_index+1) + '">\n')
        create_xml(xml_file,guess_summ_file,ref_summ_list[guess_summ_index])
        xml_file.write('</EVAL>\n')
    xml_file.write('</ROUGE-EVAL>\n')
    xml_file.close()
    
    
    # this is the file where the output of ROUGE will be stored
    ROUGE_output_path =  Dir.res+'/Temp/ROUGE_result.txt'
    
    # this is where we run ROUGE itself
    exec_command = ROUGE_path + ' -e ' + data_path + ' ' + options + ' -x ' + xml_path + ' > ' + ROUGE_output_path
    # print(exec_command)
    os.system(exec_command)
    
    # here, we read the file with the ROUGE output and
    # look for the recall, precision, and F-measure scores
    recall_list = list()
    precision_list = list()
    F_measure_list = list()
    with open(ROUGE_output_path,'r') as ROUGE_output_file:
        for n in range(ngram_order):
            ROUGE_output_file.seek(0)
            for line in ROUGE_output_file:
                match = re.findall('X ROUGE-' + str(n+1) + ' Average_R: ([0-9.]+)',line)
                if match != []:
                    recall_list.append(float(match[0]))
                match = re.findall('X ROUGE-' + str(n+1) + ' Average_P: ([0-9.]+)',line)
                if match != []:
                    precision_list.append(float(match[0]))
                match = re.findall('X ROUGE-' + str(n + 1) + ' Average_F: ([0-9.]+)', line)
                if match != []:
                    F_measure_list.append(float(match[0]))
    # with open(ROUGE_output_path, 'r') as ROUGE_output_file:
    #     content = ROUGE_output_file.read()
    #     match = re.findall('X ROUGE-L Average_R: ([0-9.]+)', content)
    #     if match != []:
    #         recall_list.append(float(match[0]))
    #     match = re.findall('X ROUGE-L Average_P: ([0-9.]+)', content)
    #     if match != []:
    #         precision_list.append(float(match[0]))
    #     match = re.findall('X ROUGE-L Average_F: ([0-9.]+)', content)
    #     if match != []:
    #         F_measure_list.append(float(match[0]))

    ROUGE_output_file.close()
    
    # remove temporary files which were created
    # os.remove(xml_path)
    # os.remove(ROUGE_output_path)

    return (recall_list,precision_list,F_measure_list)
    
    
# This is an auxiliary function
# It creates an XML file which ROUGE can read
# Don't ask me how ROUGE works, because I don't know!
def create_xml(xml_file,guess_summ_file,ref_summ_list):
    xml_file.write('<PEER-ROOT>\n')
    guess_summ_dir = os.path.dirname(guess_summ_file)
    xml_file.write(guess_summ_dir + '\n')
    xml_file.write('</PEER-ROOT>\n')
    xml_file.write('<MODEL-ROOT>\n')
    ref_summ_dir = os.path.dirname(ref_summ_list[0] + '\n')
    xml_file.write(ref_summ_dir + '\n')
    xml_file.write('</MODEL-ROOT>\n')
    xml_file.write('<INPUT-FORMAT TYPE="SPL">\n')
    xml_file.write('</INPUT-FORMAT>\n')
    xml_file.write('<PEERS>\n')
    guess_summ_basename = os.path.basename(guess_summ_file)
    xml_file.write('<P ID="X">' + guess_summ_basename + '</P>\n')
    xml_file.write('</PEERS>\n')
    xml_file.write('<MODELS>')
    letter_list = ['A','B','C','D','E','F','G','H','I','J']
    for ref_summ_index,ref_summ_file in enumerate(ref_summ_list):
        ref_summ_basename = os.path.basename(ref_summ_file)
        xml_file.write('<M ID="' + letter_list[ref_summ_index] + '">' + ref_summ_basename + '</M>\n')
    
    xml_file.write('</MODELS>\n')


def get_abs_ref_paths(abstract_dir,standard_dir):
    abs_list = os.listdir(abstract_dir)
    sta_list = os.listdir(standard_dir)
    guess_list,ref_list = [],[]
    for name in abs_list:
        if name in sta_list:
            guess_list.append(abstract_dir+name)
            ref_list.append([standard_dir+name])
    return guess_list,ref_list


def eval(abstract_dir,standard_dir,num):
    guess_summary_list = RP.get_file_path(abstract_dir)
    ref_summ_list = RP.get_file_path(standard_dir)
    guess_summary_list,ref_summ_list  = get_abs_ref_paths(abstract_dir,standard_dir)
    # print(len(ref_summ_list),len(guess_summary_list))
    #
    # print(guess_summary_list[0:5],guess_summary_list[300],guess_summary_list[-1])
    # print(ref_summ_list[0:5],ref_summ_list[300],guess_summary_list[-1])

    # print(guess_summary_list.__len__(),guess_summary_list)
    # print(ref_summ_list.__len__(),ref_summ_list)
    recall_list, precision_list, F_measure_list = PythonROUGE(guess_summary_list, ref_summ_list,ngram_order = num)
    string = ""
    string+='recall = ' + str(recall_list)+'\n'
    string += 'precision = ' + str(precision_list)+'\n'
    string += 'F = ' + str(F_measure_list) + '\n'
    # print(string)
    return string

# This is only called if this file is executed as a script.
# It shows an example of usage.
if __name__ == '__main__':
    abstract_processed = Dir.res + "/result/extradata_1000/FirstNSentencesSummarizor/abstract_processed/"
    ref = Dir.res + "/result/standard/ref_processed/"
    string = eval(abstract_dir=abstract_processed,standard_dir=ref)
    print(string)
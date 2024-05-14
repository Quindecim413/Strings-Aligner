import os
import threading
from tkinter import filedialog as fd
from tkinter import messagebox
import re
from tk_utils import options as tk_options, processing as tk_processing, validation as tk_validation
from tk_utils.progress_counter import ProgressCounter


import tkinter as tk
import pandas as pd, numpy as np



app = tk.Tk()
app.title('Выравнивание участков')
app.resizable(False, False)
app.geometry('340x260')


import psutil

selected_platform_ind = 0
n_threads = max(psutil.cpu_count(logical=True) - 2, 1)
file_short = ''
file_long = ''
file_similarity = ''
dir_res = ''
penalty_d = tk.IntVar(app, value=-5)
threshold_p = tk.DoubleVar(app, value=99.99999)
cutoff_p = threshold_p.get()
arr_short = None
arr_long = None
similarity_mat = None

one_side_expansion = 0

nmatches_total = None
nmatches = None
binom_probs = None
binom_dist = None
processing_ranges = None

compute_concolve_matches_thread: threading.Thread = None


tk.Label(app, text='Платформа\nвычислений:').place(x=10, y=10)

lb_platform = tk.Listbox(app)
for ind, pl in enumerate(tk_options.get_platform_names()):
    lb_platform.insert(ind, pl)

lb_platform.place(x=10, y=44, width=100, height=90)
def onselect_platform(*args):
    global selected_platform_ind
    selection = lb_platform.curselection()
    if selection:
        selected_platform_ind = int(selection[0])
    # value = lb_platform.get(selected_platform_ind)
    # print('Platform selected %d: "%s"' % (index, value))

lb_platform.bind('<<ListboxSelect>>', onselect_platform)
lb_platform.selection_set(selected_platform_ind)
onselect_platform()


def try_read_file(file_path):
    with open(file_path) as f:
        arr = f.read().upper()
        arr = "".join(arr.split())
    return arr



def try_update_process_ranges():
    global nmatches_total, nmatches, binom_probs, processing_ranges, binom_dist
    if arr_short is None or arr_long is None or similarity_mat is None or nmatches_total is None:
        return

    print('computing stats')
    one_side_expand_elements = int(len(arr_short) * one_side_expansion)
    # alignment_size = len(arr_short) + 2 * one_side_expand_elements
    
    p = 1/similarity_mat.shape[0]

    
    print('--0')
    print('threshold_p.get()', threshold_p.get())
    cutoff_nmatches = tk_processing.get_cutoff_val(len(arr_short), p, threshold_p.get()/100)
    print('cutoff =', cutoff_nmatches, 
            'mean =', nmatches_total.mean(),
            'median =', np.median(nmatches_total),
            'min =', nmatches_total.min(), 
            'max =', nmatches_total.max(),
            'len(arr_short) =', len(arr_short))

    print('--1')
    _nmatches, _processing_ranges = tk_processing.get_process_ranges_for_maxs(nmatches_total, cutoff_nmatches, len(arr_short), one_side_expand_elements)
    print('setup computed values')
    print('processing_ranges', len(_processing_ranges))
    
    
    print('--2')
    binom_dist = tk_processing.get_binomial_pmf(len(arr_short), p)
    
    _binom_probs = binom_dist[_nmatches]
    
    nmatches = _nmatches
    processing_ranges = _processing_ranges
    binom_probs = _binom_probs
    binom_dist = binom_dist

    cutoff_p = threshold_p.get()
    lbl_num_tasks.configure(text='{}'.format(len(processing_ranges)))


def show_warning_if_invalid_ranges_exist():
    num_neg1 = (nmatches_total == -1).sum()

    if num_neg1 > 0:
        _, diff_arrlong, _ = tk_validation.check_two_arrays_simbols(arr_long, similarity_mat.columns)
        print('diff_arrlong', diff_arrlong)
        messagebox.showwarning(title='Неопознанные символы', message='''Длинный массив содержит символы,\nкоторых нет в матрице сравнений:\n{}
Таким образом, {} промежутков выравнивания\nбудет проигнорировано!'''.format(
                                '"'+'", "'.join(diff_arrlong)+'"',
                                num_neg1
                            ))


def chech_short_arr_and_smat_simbols_match(shortarr, smat_simbols):
    _, diffshort, _ = tk_validation.check_two_arrays_simbols(shortarr, smat_simbols)
    if diffshort:
        messagebox.showerror(title='Неопознанные символв', message='Короткий массив содержит символы,\nкоторых нет в матрице сравнений:\n{}'.format(
            '"'+'", "'.join(diffshort)+'"'
        ))
        return False
    return True

def try_compute_convolve_nmatches():
    if arr_short is None or arr_long is None or similarity_mat is None:
        return
    
    def do():
        print('convoling nmatches')
        mapping = tk_processing.get_forward_mapping(similarity_mat)
        _nmatches = tk_processing.convolve_nmatches(tk_processing.encode_arr(arr_short, mapping), tk_processing.encode_arr(arr_long, mapping), selected_platform_ind)
        global nmatches_total
        nmatches_total = _nmatches
        print('quit thread compute_concolve_matches_thread')
        

    global compute_concolve_matches_thread
    
    print('before init compute_concolve_matches_thread')
    compute_concolve_matches_thread = threading.Thread(target=do)
    compute_concolve_matches_thread.daemon = True
    compute_concolve_matches_thread.start()
    check_compute_convolve_matches()
    print('after init compute_concolve_matches_thread')

    
    btn_path_short.config(state=tk.DISABLED)
    btn_path_long.config(state=tk.DISABLED)
    btn_path_similarity.config(state=tk.DISABLED)
    btn_process.config(state=tk.DISABLED)
    btn_update_num_jobs.config(state=tk.DISABLED)


ind_compute_convolve_nmatches = 0
def check_compute_convolve_matches():
    global ind_compute_convolve_nmatches
    if compute_concolve_matches_thread.is_alive():
        ind_compute_convolve_nmatches += 1
        lbl_num_tasks.configure(text='Расчёт'+'.'*(ind_compute_convolve_nmatches%6+1))
        app.after(200, check_compute_convolve_matches)
    else:
        btn_path_short.config(state=tk.NORMAL)
        btn_path_long.config(state=tk.NORMAL)
        btn_path_similarity.config(state=tk.NORMAL)
        btn_process.config(state=tk.NORMAL)
        btn_update_num_jobs.config(state=tk.NORMAL)
        print('check_compute_convolve_matches calls try_update_process_ranges')
        try_update_process_ranges()
        show_warning_if_invalid_ranges_exist()


def select_short():
    global file_short
    filetypes = (
        ('txt file', '*.txt'),
    )
    filename = fd.askopenfilename(
        title='Короткий файл',
        initialdir='.',
        filetypes=filetypes)
    
    if filename:
        
        global arr_short
        try:
            arr = try_read_file(filename)
        except Exception as e:
            messagebox.showerror('', 'Не удалось открыть файл:{}\n{}'.format(filename, e))
            return
        if arr_long:
            if not tk_validation.check_arrshort_and_arrlong_length(len(arr), len(arr_long)):
                messagebox.showerror(title='Несоответсвие длин строк', message='Длина короткой строки должна быть\nменьше или равна длинной')
                return
        if '-' in arr:
            messagebox.showerror(title='Недопустимый символ', message='Запрещено наличие в файле с последовательностью\nсимвола "-"')
            return
        
        if len(arr) == 0:
            messagebox.showerror(title='Нет данных', message='Файл не содержит символов последовательностей')
            return

        if similarity_mat is not None:
            if not chech_short_arr_and_smat_simbols_match(arr, similarity_mat):
                return
        arr_short = arr
        file_short = filename
        update_mem_usage_lbl()
        try_compute_convolve_nmatches()
        
        

def select_long():
    global file_long
    filetypes = (
        ('txt file', '*.txt'),
    )
    filename = fd.askopenfilename(
        title='Длинный файл',
        initialdir='.',
        filetypes=filetypes)
    
    if filename:
        
        global arr_long
        try:
            arr = try_read_file(filename)
        except Exception as e:
            messagebox.showerror('', 'Не удалось открыть файл:{}\n{}'.format(filename, e))
            return
        if arr_short:
            if not tk_validation.check_arrshort_and_arrlong_length(len(arr_short), len(arr)):
                messagebox.showerror(title='Несоответсвие длин строк', message='Длина длинной строки должна быть\nбольше или равна короткой')
                return

        if '-' in arr:
            messagebox.showerror(title='Недопустимый символ', message='Запрещено наличие в файле с последовательностью\nсимвола "-"')
            return
        
        if len(arr) == 0:
            messagebox.showerror(title='Нет данных', message='Файл не содержит символов последовательностей')
            return

        arr_long = arr
        file_long = filename
        try_compute_convolve_nmatches()

def select_similarity():
    global file_similarity
    filetypes = (
        ('Excel', '*.xlsx'),
        ('CSV, разделитель табуляция', '*.csv'),
    )

    filename = fd.askopenfilename(
        title='Файл с матрицей сравнений',
        initialdir='.',
        filetypes=filetypes)
    
    if filename:
        
        global similarity_mat
        try:
            s_1 = pd.read_excel(filename, index_col=0)
            s_1 = s_1.dropna(axis=0, how='all').dropna(axis=1, how='all')
            s = s_1.fillna(0)
            filled_na_values = (s_1.values != s.values).any()
        except Exception as e:
            messagebox.showerror('', 'Не удалось открыть файл:{}\n{}'.format(filename, e))
            return
        cols = list(map(lambda el: str(el).upper(), s.columns))
        rows = list(map(lambda el: str(el).upper(), s.index))
        
        if len(cols) == 0:
            messagebox.showerror(title='Нет данных', message='Файл с матрицей соответствий пуст!')
            return

        found_empty = False
        for col in cols:
            found_items = re.findall('\s', col)
            if len(found_items) > 0:
                found_empty = True
                break
        
        if found_empty:
            messagebox.showerror(title='Неверный формат матрицы соответствий', message='Не допускается использование\nневидимых символов в качестве\nназвание элемента матрицы')
            return

        if '-' in cols:
            messagebox.showerror(title='Неверный формат матрицы соответствий', message='Не допускается использование символа "-"\nв качестве название элемента матрицы')
            return

        ok, diff_cols, diff_rows =tk_validation.check_two_arrays_simbols(cols, rows)
        if not ok or diff_cols or diff_rows:
            messagebox.showerror(title='Неверный формат матрицы соответствий', message='Названия символов в матрице сопоставлений\nне должны дублироваться, а также\nдолжны совпадаться по столбцам и строкам')
            return

        if not tk_validation.check_smat_simbols_lengths(cols):
            messagebox.showerror(title='Неверный формат матрицы соответствий', message='Названия символов в матрице сопоставлений\nдолжны быть одноэлементными')
            return


        s = tk_processing.organize_smat_rows_and_columns(s)

        if arr_short is not None:
            if not chech_short_arr_and_smat_simbols_match(arr_short, s):
                return
            
        if filled_na_values:
            messagebox.showwarning(title='Уведомление', message='В загруженной матрице соответвий\nбыли найдены пустые ячейки.\nИх значение приняты за 0')

        file_similarity = filename
        similarity_mat = pd.DataFrame(s.values.astype('int32'), index=s.index, columns=s.columns)
        print('similarity matrix')
        print(similarity_mat)
        try_compute_convolve_nmatches()


def select_results():
    global dir_res

    dirname = fd.askdirectory(
        title='Итоговые результаты',
        initialdir='.')
    
    if dirname:
        dir_res = dirname


tk.Label(app, text='Файлы данных:').place(x=110, y=10)
btn_path_short = tk.Button(app, text='Короткий файл', command=select_short)
btn_path_long  = tk.Button(app, text='Длинный файл', command=select_long)
btn_path_similarity = tk.Button(app, text='Файл сравнений', command=select_similarity)
btn_path_results  = tk.Button(app, text='Папка результатов', command=select_results)

btn_path_short.place(x=110, y=30, width=110, height=25)
btn_path_long.place(x=220, y=30, width=110, height=25)
btn_path_similarity.place(x=110, y=55, width=110, height=25)
btn_path_results.place(x=220, y=55, width=110, height=25)


def show_files():
    templ = "Короткий файл:\n{}\nДлинный файл:\n{}\nФайл с матрицей сравнений:\n{}\nПапка результатов:\n{}"
    res = templ.format(
        file_short if file_short else "НЕ ВЫБРАН",
        file_long if file_long else "НЕ ВЫБРАН",
        file_similarity if file_similarity else "НЕ ВЫБРАН",
        dir_res if dir_res else "НЕ ВЫБРАН"
    )
    messagebox.showinfo("Пути к файлам",res)

btn_show_paths = tk.Button(app, text='Показать выбранные файлы', command=show_files)
btn_show_paths.place(x=110, y=80, width=220, height=25)

def callback_validate_d(P):
    import re
    # print('validate', P)
    m = re.fullmatch("-[0-9]{1,3}", P)
    # print(m)
    return m is not None

vcmd_d = (app.register(callback_validate_d))

tk.Label(app, text='d =').place(x=110, y=110)
d_spinbox = tk.Spinbox(app, from_=-999, to=0, validate='key', textvariable=penalty_d, validatecommand=(vcmd_d, '%P'), command=lambda: print(penalty_d.get()))
d_spinbox.place(x=140, y=112, width=50)

def callback_validate_p(P):
    import re
    # print('validate', P)
    m = re.fullmatch("[0-9]{1,2}(\\.[0-9]{0,9})?", P)
    # print(m)
    return m is not None

vcmd_p = (app.register(callback_validate_p))
tk.Label(app, text='p = ').place(x=200, y=110)
p_spinbox = tk.Spinbox(app, from_=0, to=100, textvariable=threshold_p, format='%.9f', increment=0.001, 
            validate='key',  validatecommand=(vcmd_p, '%P'),
            )
p_spinbox.place(x=225, y=112, width=87)
tk.Label(app, text='%').place(x=315, y=110)

btn_update_num_jobs = tk.Button(app, text='Обновить', command=try_update_process_ranges)
btn_update_num_jobs.place(x=273, y=139, height=20)

tk.Label(app, text='К-во промежутков выравнивания:').place(x=10, y=140)
lbl_num_tasks = tk.Label(app, text='-')
lbl_num_tasks.place(x=203, y=140)





def update_mem_usage_lbl():
    global arr_short, n_threads
    if arr_short is not None:
        usage_per_thr = tk_options.get_max_mem_usage_per_thread(int(len(arr_short) * 1.2))
    else:
        usage_per_thr = 0
    usage_mb = round((usage_per_thr * n_threads) / (2**20), 3)
    mem_load_lbl.configure(text="Ориентировочное использование памяти: {} Мб".format(usage_mb))

mem_load_lbl = tk.Label(app)
mem_load_lbl.place(x=10, y=170)


def callback_validate(P):
    global n_threads
    if str.isdigit(P):
        el = int(P)
        if el > 20 or el < 1:
            return False
        n_threads = el
        update_mem_usage_lbl()
        return True
    elif P == "":
        n_thr_var.set(str(n_threads))
        return True
    else:
        return False

tk.Label(app, text='К-во потоков').place(x=10, y=200)

vcmd = (app.register(callback_validate))

n_thr_var = tk.StringVar(value=str(n_threads))
nthreads_entry = tk.Entry(app, validate='key', validatecommand=(vcmd, '%P'), textvariable=n_thr_var) 
nthreads_entry.place(x=100, y=200, width=20)



counter: ProgressCounter = None
process_arrays_thread: threading.Thread = None
processing_exception = None
write_files_ind = 1
saving_files = False
def progress_counter_show_message():
    global counter, process_arrays_thread, write_files_ind, saving_files, err_save
    
    
    if err_save:
        process_arrays_thread = None
        messagebox.showerror(title='Ошибка сохранения', message=err_save)
        err_save = None
        btn_path_short.config(state=tk.NORMAL)
        btn_path_long.config(state=tk.NORMAL)
        btn_path_similarity.config(state=tk.NORMAL)
        btn_path_results.config(state=tk.NORMAL)
        
        d_spinbox.config(state=tk.NORMAL)
        p_spinbox.config(state=tk.NORMAL)
        
        btn_process.config(state=tk.NORMAL)
        btn_update_num_jobs.config(state=tk.NORMAL)
        
        return

    res = "{}/{}".format(counter.processed, counter.max_val)
    if process_arrays_thread.is_alive():
        if saving_files:
            write_files_ind += 1
            res = 'Пишем файлы' + '.'*(write_files_ind%6)
        lbl_progress_process.configure(text=res)
        app.after(200, progress_counter_show_message)
    else:
        lbl_progress_process.configure(text=res + ' Готово!')
        
        process_arrays_thread = None

        btn_path_short.config(state=tk.NORMAL)
        btn_path_long.config(state=tk.NORMAL)
        btn_path_similarity.config(state=tk.NORMAL)
        btn_path_results.config(state=tk.NORMAL)
        
        d_spinbox.config(state=tk.NORMAL)
        p_spinbox.config(state=tk.NORMAL)
        
        btn_process.config(state=tk.NORMAL)
        btn_update_num_jobs.config(state=tk.NORMAL)

err_save = None
def process_data():
    print(penalty_d.get())
    global nmatches, binom_probs, processing_ranges
    global arr_short, arr_long, similarity_mat
    global n_threads
    global counter, process_arrays_thread

    if process_arrays_thread is not None:
        messagebox.showerror('Идет обработка', message='В данный момент идет обработка данных')
        return

    if nmatches is None or binom_probs is None or processing_ranges is None or arr_short is None or arr_long is None or similarity_mat is None:
        messagebox.showerror('Нет данных', message='Сначала загрузите данные для обработки')
        return

    if not os.path.isdir(dir_res):
        messagebox.showerror('Неверный путь', message='Сначала укажите существующую папку\nдля сохрания результатов')
        return

    btn_path_short.config(state=tk.DISABLED)
    btn_path_long.config(state=tk.DISABLED)
    btn_path_similarity.config(state=tk.DISABLED)
    btn_path_results.config(state=tk.DISABLED)
    
    d_spinbox.config(state=tk.DISABLED)
    p_spinbox.config(state=tk.DISABLED)

    btn_process.config(state=tk.DISABLED)
    btn_update_num_jobs.config(state=tk.DISABLED)
    
    counter = ProgressCounter(0, len(processing_ranges))
    def do_job(counter=counter):
        try:
            d = penalty_d.get()
            global err_save, saving_files
            err_save = None
            saving_files = True
            try:
                if not os.path.exists(dir_res) and not os.path.isdir(dir_res):
                    from pathlib import Path
                    path = Path(dir_res)
                    path.mkdir(parents=True)
            except:
                err_save = 'Не удалось инициализировать\nпапку для сохранения результатов\n{}'.format(dir_res)
                return
            try:
                file_save = os.path.join(dir_res, 'Входные параметры алгоритма выравнивания.xlsx')
                print('write', file_save)
                pd.Series({
                    'Короткий файл': file_short,
                    'Длинный файл': file_long,
                    'Матрица сравнений': file_similarity,
                    'Папка результатов': dir_res,
                    'd': d,
                    'p': cutoff_p,
                    'Длина короткой последовательности': len(arr_short),
                    'Длина длинной последовательности': len(arr_long),
                    'К-во потоков': n_threads,
                    'К-во промежутков выравнивания на обработку': len(processing_ranges),
                    'К-во игнорируемых промежутков выравнивания': (nmatches_total == -1).sum(),
                    'HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE': tk_options.get_options().HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE,
                    'NW_ALIGN_MINLEN_FOR_GPU': tk_options.get_options().NW_ALIGN_MINLEN_FOR_GPU,
                    'NW_SCORE_MINLEN_FOR_GPU': tk_options.get_options().NW_SCORE_MINLEN_FOR_GPU
                }).to_excel(file_save, header=None)
            except:
                err_save = 'Не удалось произвести запись в файл\n{}'.format(file_save)
                return
            try:
                file_save = os.path.join(dir_res,'Вероятности по биномиальному распределению.csv')
                print('write', file_save)
                pd.Series(binom_probs).to_csv(file_save, index=False, header=None)
            except:
                err_save = 'Не удалось произвести запись в файл\n{}'.format(file_save)
                return
            try:
                file_save = os.path.join(dir_res,'Биномиальное распределение.csv')
                print('write', file_save)
                pd.Series(binom_dist).to_csv(file_save, index=False, header=None)
            except:
                err_save = 'Не удалось произвести запись в файл\n{}'.format(file_save)
                return
            try:
                file_save = os.path.join(dir_res, 'К-во совпадений.csv')
                print('write', file_save)
                with open(file_save, 'w') as f:
                    step = 100_000
                    i = 0
                    while i < len(nmatches_total):
                        print('write sub of ntotal', i, '->', min(i+step, len(nmatches_total)))
                        sub_str = '\n'.join(map(str, nmatches_total[i:i+step]))
                        f.write(sub_str)
                        f.write('\n')
                        f.flush()
                        i += step
                # df_matches = pd.Series(nmatches_total, name='К-во совпадений').to_frame()
                # # df_matches['Доля совпадений'] = np.where(nmatches_total >= 0, nmatches_total/len(arr_short), -1)
                # df_matches.to_csv(file_save, index=False, header=None)
            except:
                err_save = 'Не удалось произвести запись в файл\n{}'.format(file_save)
                return

            saving_files = False
            print('start tk_processing.process_arrays')
            err_files = tk_processing.process_arrays(
                arr_short, arr_long, similarity_mat, d,
                processing_ranges, nmatches, binom_probs,
                n_threads, selected_platform_ind, dir_res, counter
            )
            if err_files:
                err_save = 'Не удалось произвести запись в файл(-ы)\n{}'.format(err_files)
                return

            
    #         binom_probs = _binom_probs
    # binom_dist = binom_dist
        except Exception as e:
            raise e

    # def do_check():
    #     a_set = set(a)

    print('before init process_arrays_thread')
    process_arrays_thread = threading.Thread(target=do_job)
    process_arrays_thread.setDaemon(True)
    process_arrays_thread.start()
    print('after init process_arrays_thread')
    progress_counter_show_message() 



btn_process = tk.Button(app, text='Обработать', command=process_data, state=tk.DISABLED)
btn_process.place(x=10, y=230)

tk.Label(app, text='Статус обработки: ').place(x=90, y=234)
lbl_progress_process = tk.Label(app, text='-')
lbl_progress_process.place(x=200, y=234)

def try_load_config():
    err = tk_options.load_config_from_file()
    if err:
        messagebox.showwarning("Проблема загрузки файла конфигурации", err)
app.after(250, try_load_config)
app.mainloop()

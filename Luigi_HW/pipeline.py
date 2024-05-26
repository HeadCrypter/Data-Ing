# Прокидываем библиотеки
import os
import luigi
import subprocess
import gzip
import shutil
import pandas as pd
import io
import logging
from datetime import datetime
from bs4 import BeautifulSoup
import requests

# Заводим логгирование
logger = logging.getLogger('luigi-interface')
logging.basicConfig(level=logging.INFO)

# Формируем ссылку для скачивания архива
def get_download_url(dataset_series, dataset_name):
    base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{dataset_series}/{dataset_name}/suppl/"
    page = requests.get(base_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    for link in soup.find_all('a'):
        if 'RAW.tar' in link.get('href'):
            return base_url + link.get('href')
    raise ValueError("Ссылка на архив не найдена")

# Задаем задачу для скачивания в папку data
class GeoLoader(luigi.Task):
    data_dir = luigi.Parameter(default='data')
    dataset_series = luigi.Parameter(default='GSE68nnn')
    dataset_name = luigi.Parameter(default='GSE68849')

    def output(self):
        # Прокидываем путь к скачанному файлу
        return luigi.LocalTarget(os.path.join(self.data_dir, f"{self.dataset_name}_RAW.tar"))

    def run(self):
        # Создаем директорию, если она не существует
        os.makedirs(self.data_dir, exist_ok=True)
        # Получаем URL для скачивания архива
        download_url = get_download_url(self.dataset_series, self.dataset_name)
        output_path = self.output().path

        # Скачиваем архив с помощью wget
        subprocess.run(["wget", "-c", "-O", output_path, download_url], check=True)
        logger.info(f"Файл скачан по пути: {output_path}")

# Проводим разархивацию файлов
class GzipExtracter(luigi.Task):
    data_dir = luigi.Parameter(default='data')
    dataset_name = luigi.Parameter(default='GSE68849')

    def requires(self):
        # Зависит от GeoLoader
        return GeoLoader(data_dir=self.data_dir, dataset_series='GSE68nnn', dataset_name=self.dataset_name)

    def output(self):
        # Указываем путь к директории, где будут распакованы файлы
        return luigi.LocalTarget(os.path.join(self.data_dir, self.dataset_name, 'extracted'))

    def run(self):
        tar_path = self.input().path
        extract_path = self.output().path
        os.makedirs(extract_path, exist_ok=True)

        # Распаковываем tar-архив
        subprocess.run(["tar", "-xvf", tar_path, "-C", extract_path], check=True)
        logger.info(f"Распакован tar-архив в: {extract_path}")

        # Распаковываем каждый gzip-файл внутри tar-архива
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.gz'):
                    gzip_file_path = os.path.join(root, file)
                    output_file_path = os.path.splitext(gzip_file_path)[0]

                    with gzip.open(gzip_file_path, 'rb') as f_in, open(output_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(gzip_file_path)
                    logger.info(f"Распакован gzip-файл в: {output_file_path}")

class TextReducer(luigi.Task):
    data_dir = luigi.Parameter(default='data')
    dataset_name = luigi.Parameter(default='GSE68849')

    def requires(self):
        # Зависит от GzipExtracter
        return GzipExtracter(data_dir=self.data_dir, dataset_name=self.dataset_name)

    def output(self):
        # Указываем пути к выходным файлам для каждой таблицы
        tables = ['Heading', 'Probes', 'Controls', 'Columns', 'Probes_reduced']
        return [luigi.LocalTarget(os.path.join(self.data_dir, self.dataset_name, table, f"{table}.tsv")) for table in tables]

    def run(self):
        extract_path = os.path.join(self.data_dir, self.dataset_name, 'extracted')
        os.makedirs(os.path.join(self.data_dir, self.dataset_name), exist_ok=True)

        # Обрабатываем каждый текстовый файл
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.txt'):
                    self.process_file(os.path.join(root, file))

    def process_file(self, file_path):
        dfs = {}
        with open(file_path, 'r') as f:
            write_key = None
            fio = io.StringIO()
            for line in f:
                if line.startswith('['):
                    if write_key:
                        fio.seek(0)
                        header = None if write_key == 'Heading' else 'infer'
                        dfs[write_key] = pd.read_csv(fio, sep='\t', header=header)
                    fio = io.StringIO()
                    write_key = line.strip('[]\n')
                    continue
                if write_key:
                    fio.write(line)
            fio.seek(0)
            dfs[write_key] = pd.read_csv(fio, sep='\t')

        # Сохраняем каждую таблицу в отдельный файл
        for key, df in dfs.items():
            output_dir = os.path.join(self.data_dir, self.dataset_name, key)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{key}.tsv")
            df.to_csv(output_file, sep='\t', index=False)
            logger.info(f"Сохранена таблица {key} в файл {output_file}")

        # Обрабатываем таблицу Probes отдельно
        if 'Probes' in dfs:
            self.process_probes(dfs['Probes'])

    def process_probes(self, probes_df):
        # Удаляем ненужные колонки из таблицы Probes
        columns_to_drop = ['Definition', 'Ontology_Component', 'Ontology_Process', 'Ontology_Function', 'Synonyms', 'Obsolete_Probe_Id', 'Probe_Sequence']
        reduced_probes_df = probes_df.drop(columns=columns_to_drop)
        output_dir = os.path.join(self.data_dir, self.dataset_name, 'Probes_reduced')
        os.makedirs(output_dir, exist_ok=True)
        reduced_probes_path = os.path.join(output_dir, 'Probes_reduced.tsv')
        reduced_probes_df.to_csv(reduced_probes_path, sep='\t', index=False)
        logger.info(f"Урезанная версия таблицы Probes сохранена в: {reduced_probes_path}")

class MessCleaner(luigi.Task):
    data_dir = luigi.Parameter(default='data')
    dataset_name = luigi.Parameter(default='GSE68849')

    def requires(self):
        # Зависит от TextReducer
        return TextReducer(data_dir=self.data_dir, dataset_name=self.dataset_name)

    def output(self):
        # Указываем путь к файлу readme.txt
        return luigi.LocalTarget(os.path.join(self.data_dir, self.dataset_name, 'readme.txt'))

    def run(self):
        extract_path = os.path.join(self.data_dir, self.dataset_name, 'extracted')
        removed_files = []

        # Удаляем уже лишние исходные текстовые файлы
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    removed_files.append(file_path)
                    logger.info(f"Удален исходный текстовый файл: {file_path}")

        # Чистим от директории extracted
        shutil.rmtree(extract_path)
        logger.info(f"Директория {extract_path} удалена")

        # Создаем файл readme.txt с информацией о выполненных шагах
        readme_path = self.output().path
        with open(readme_path, 'w') as readme_file:
            readme_file.write("Пайплайн отработал без ошибок\n\n")
            readme_file.write("Этапы выполнения ------------------------\n")
            readme_file.write("1) Подгрузка архива\n")
            readme_file.write("2) Разархивация\n")
            readme_file.write("3) Процессинг текстовых файлов\n")
            readme_file.write("4) Очистка от исходных текстовых файлов\n")
            readme_file.write("5) Удаление директории extracted\n\n")
            readme_file.write("### Удаленные файлы ###\n")
            for file_path in removed_files:
                readme_file.write(f"{file_path}\n")

        logger.info(f"Создан readme файл: {readme_path}")

class AllRuner(luigi.WrapperTask):
    data_dir = luigi.Parameter(default='data')
    dataset_series = luigi.Parameter(default='GSE68nnn')
    dataset_name = luigi.Parameter(default='GSE68849')

    def requires(self):
        # Зависит от MessCleaner
        return MessCleaner(data_dir=self.data_dir, dataset_name=self.dataset_name)

if __name__ == '__main__':
    luigi.run()

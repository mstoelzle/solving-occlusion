import pathlib

import gspread
import datetime
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

from ..log import ROOT_DIR, get_logger
from .requests import create_requests
from .gather_results import gather_results_df, COLUMNS

SCOPE = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
CREDENTIALS_FILEPATH = pathlib.Path('/home/idscadmin/secondorderlearning-f737ea086b4b.json')
CREDENTIALS = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILEPATH, SCOPE)
SPREADSHEET = "SecondOrderLearning"


class SheetUploader:

    def __init__(self, experiment_name):

        self.experiment_name = experiment_name
        self.logger = get_logger(f"sheets_uploader")

        self.worksheet = None
        self.spreadsheet = None

        # Connect to google sheets
        self.connect()

    def create_column_headings(self):
        # Initialize and stylize Google sheets
        cell_list = self.worksheet.range(1, 1, 1, len(COLUMNS))
        for cell in cell_list:
            cell.value = COLUMNS[cell.col-1]

        self.worksheet.update_cells(cell_list)
        requests = create_requests(self.worksheet.id)
        self.spreadsheet.batch_update(body=requests)

    def connect(self):
        # Need to refresh connection to Google sheets, otherwise credentials are rejected after a certain time
        credentials = gspread.authorize(CREDENTIALS)
        self.spreadsheet = credentials.open(SPREADSHEET)
        if self.experiment_name not in [entry.title for entry in self.spreadsheet.worksheets()]:
            self.create_worksheet()
        else:
            self.worksheet = self.spreadsheet.worksheet(self.experiment_name)

    def create_worksheet(self):
        n_cols: str = str(2 * len(COLUMNS))
        self.worksheet = self.spreadsheet.add_worksheet(title=self.experiment_name, rows="5", cols=n_cols)

    def clean_sheet(self):
        cell_list = self.worksheet.range(2, 1, self.worksheet.row_count, len(COLUMNS))
        for cell in cell_list:
            cell.value = ""
        self.worksheet.update_cells(cell_list)

    def create_rows(self, n_rows: int):
        row_count = self.worksheet.row_count
        # offset by one to account for the header row
        n_new_rows = n_rows - row_count + 1
        if n_new_rows > 0:
            self.worksheet.add_rows(n_new_rows)

    def upload(self):
        data: pd.DataFrame = gather_results_df([self.experiment_name])
        if len(data) > 0:
            self.connect()
            self.worksheet.clear()
            self.logger.info(f"Uploading data with length {len(data)}")
            self.create_rows(len(data)+5)
            self.create_column_headings()

            cell_list = self.worksheet.range(2, 1, len(data) + 1, len(COLUMNS))
            for cell in cell_list:
                # offset rows by 2 to account for 1 indexing in gspread and the header row
                # offset cols by 1 to account for 1 indexing in gspread.

                cell.value = data.iloc[cell.row - 2][data.columns[cell.col - 1]]
                try:
                    # handle numpy items
                    cell.value = cell.value.item()
                except AttributeError:
                    pass

            self.worksheet.update_cells(cell_list)
            self.refresh_last_updated_cell()
        else:
            self.logger.critical(f"Found no data for experiment <{self.experiment_name}>")

    def refresh_last_updated_cell(self):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y/%m/%d")
        time_str = now.strftime("%H:%M:%S")
        self.worksheet.update_cell(row=1, col=len(COLUMNS) + 1, value=f"Last update {date_str} - {time_str}")

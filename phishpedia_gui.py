import sys
from PyQt5.QtWidgets import QApplication
from GUItool.ui import PhishpediaUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PhishpediaUI()
    window.show()
    sys.exit(app.exec_())

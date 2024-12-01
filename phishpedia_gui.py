import sys
from PyQt5.QtWidgets import QApplication
from GUI.ui import PhishpediaUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PhishpediaUI()
    ex.show()
    sys.exit(app.exec_())

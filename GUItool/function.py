from PyQt5.QtWidgets import (
    QMessageBox, QFileDialog, QTreeWidgetItem, QDialog, QVBoxLayout, QLabel
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import os
import shutil
import pickle
from configs import load_config
from phishpedia import PhishpediaWrapper


class PhishpediaFunction:
    def __init__(self, ui):
        self.ui = ui
        self.phishpedia_cls = PhishpediaWrapper()
        self.current_pixmap = None

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self.ui, "Select Screenshot", "", "Images (*.png *.jpg *.jpeg)",
                                                   options=options)
        if file_name:
            self.ui.image_input.setText(file_name)

    def detect_phishing(self):
        url = self.ui.url_input.text()
        screenshot_path = self.ui.image_input.text()

        if not url or not screenshot_path:
            self.ui.category_display.setText("Please enter URL and upload a screenshot.")
            self.ui.target_display.clear()
            self.ui.domain_display.clear()
            return

        phish_category, pred_target, matched_domain, plotvis, siamese_conf, pred_boxes, logo_recog_time, logo_match_time = self.phishpedia_cls.test_orig_phishpedia(
            url, screenshot_path, None)

        # 设置检测结果类别和颜色
        if phish_category == 0:
            self.ui.category_display.setStyleSheet("color: green;")
            self.ui.category_display.setText("Benign")
        elif phish_category == 1:
            self.ui.category_display.setStyleSheet("color: red;")
            self.ui.category_display.setText("Phish")
        
        # 如果没有匹配到目标，显示黄色的No match
        if pred_target is None or pred_target == "":
            self.ui.category_display.setStyleSheet("color: orange;")
            self.ui.category_display.setText("No Match")
            pred_target = "None"
        
        # 更新其他显示框的内容
        self.ui.target_display.setText(str(pred_target))
        self.ui.domain_display.setText(str(matched_domain))

        if phish_category == 1 and plotvis is not None:
            self.display_image(plotvis)
        if phish_category == 0:
            self.display_image(plotvis)

    def display_image(self, plotvis):
        try:
            # Convert BGR to RGB
            plotvis_rgb = cv2.cvtColor(plotvis, cv2.COLOR_BGR2RGB)
            height, width, channel = plotvis_rgb.shape
            bytes_per_line = 3 * width
            plotvis_qimage = QImage(plotvis_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            self.current_pixmap = QPixmap.fromImage(plotvis_qimage)
            self.update_image_display()
        except Exception as e:
            print(f"Error converting image: {e}")

    def update_image_display(self):
        if self.current_pixmap:
            # Get the actual size of the visualization_display
            display_height = self.ui.visualization_display.height()
            display_width = self.ui.visualization_display.width()
            # Get the original dimensions of the image
            original_width = self.current_pixmap.width()
            original_height = self.current_pixmap.height()
            # Calculate the scaling ratio
            width_ratio = display_width / original_width
            height_ratio = display_height / original_height
            # Use the smaller ratio to ensure the image fits completely within the display area
            scale_ratio = min(width_ratio, height_ratio)
            # Calculate the scaled dimensions
            new_width = int(original_width * scale_ratio)
            new_height = int(original_height * scale_ratio)
            # Scale the image
            scaled_pixmap = self.current_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.visualization_display.setPixmap(scaled_pixmap)

    def on_resize(self, event):
        self.update_image_display()

    def get_directory_structure(self, path):
        directory_structure = {}
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                directory_structure[item] = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
        return directory_structure

    def on_item_clicked(self, item, column):
        # Check if it's a logo file (child item)
        if item.parent() is not None:  # Only for logo files (child items)
            logo_path = f"models/expand_targetlist/{item.parent().text(0)}/{item.text(0)}"
            if logo_path.endswith('.png'):
                self.show_logo_image(logo_path)

    def show_logo_image(self, logo_path):
        try:
            image = QImage(logo_path)
            if image.isNull():
                QMessageBox.warning(self.ui, "Warning", f"Failed to load image: {logo_path}")
                return

            # Scale image if it's too large
            max_width = 800
            max_height = 600
            if image.width() > max_width or image.height() > max_height:
                image = image.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            pixmap = QPixmap.fromImage(image)

            # Create dialog with a title showing the logo name
            dialog = QDialog(self.ui)
            dialog.setWindowTitle(f"Logo Image - {os.path.basename(logo_path)}")
            
            # Set dialog size based on image size plus padding
            dialog.resize(pixmap.width() + 40, pixmap.height() + 40)
            
            # Center the image in the dialog
            dialog_layout = QVBoxLayout()
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            dialog_layout.addWidget(image_label)
            
            dialog.setLayout(dialog_layout)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Error displaying image: {str(e)}")

    def populate_tree(self, tree_widget, directory_structure):
        for brand, logos in directory_structure.items():
            brand_item = QTreeWidgetItem([brand])
            for logo in logos:
                logo_item = QTreeWidgetItem([logo])
                brand_item.addChild(logo_item)
            tree_widget.addTopLevelItem(brand_item)

    def add_brand(self):
        # Create dialog using UI method
        dialog, brand_input, domain_input, add_btn, cancel_btn = self.ui.create_add_brand_dialog(self)
        
        # Button connections
        def on_add():
            brand_name = brand_input.text().strip()
            domains = domain_input.text().strip()
            
            # Validate brand name
            if not brand_name:
                QMessageBox.warning(
                    dialog,
                    "Warning",
                    "Brand name is required!"
                )
                return
            
            if not all(c.isalnum() or c.isspace() or c in '-_' for c in brand_name):
                QMessageBox.warning(
                    dialog,
                    "Warning",
                    "Brand name can only contain letters, numbers, spaces, hyphens and underscores!"
                )
                return
            
            # Validate domains
            if not domains:
                QMessageBox.warning(dialog, "Warning", "Domain name is required!")
                return
            
            # Create brand directory
            brand_path = os.path.join('models/expand_targetlist', brand_name)
            try:
                if not os.path.exists(brand_path):
                    os.makedirs(brand_path)
                    # Update tree view
                    brand_item = QTreeWidgetItem([brand_name])
                    self.ui.tree_widget.addTopLevelItem(brand_item)
                    
                    # Update domain mapping
                    if self.domain_map_add(brand_name, domains):
                        QMessageBox.information(
                            dialog,
                            "Success",
                            "Brand and domains added successfully!\nPlease click 'Reload Model' button to reload the models."
                        )
                        dialog.accept()
                else:
                    QMessageBox.warning(dialog, "Warning", "Brand already exists!")
            except Exception as e:
                QMessageBox.critical(dialog, "Error", f"Failed to create brand: {str(e)}")
        
        def on_cancel():
            dialog.reject()
        
        add_btn.clicked.connect(on_add)
        cancel_btn.clicked.connect(on_cancel)
        
        # Show dialog
        dialog.exec_()

    def delete_brand(self):
        # Get selected item
        selected_item = self.ui.tree_widget.currentItem()
        
        if selected_item and not selected_item.parent():  # Ensure it's a brand (top-level directory)
            brand_name = selected_item.text(0)
            brand_path = os.path.join('models/expand_targetlist', brand_name)
            
            # Protect root directory
            if brand_path == 'models/expand_targetlist':
                QMessageBox.warning(self.ui, "Warning", "Cannot delete root directory!")
                return

            # Confirm deletion
            reply = QMessageBox.question(
                self.ui,
                'Confirm Delete',
                f'Are you sure you want to delete brand "{brand_name}" and all its logos?\n'
                'This will also delete the corresponding domain mapping.',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    # Delete directory and its contents
                    shutil.rmtree(brand_path)
                    # Remove from tree view
                    self.ui.tree_widget.takeTopLevelItem(
                        self.ui.tree_widget.indexOfTopLevelItem(selected_item)
                    )
                    
                    # Update domain mapping
                    self.domain_map_delete(brand_name)
                    
                    QMessageBox.information(
                        self.ui,
                        "Success",
                        "Brand and domains deleted successfully!\n"
                        "Please click 'Reload Model' button to reload the models."
                    )
                except Exception as e:
                    QMessageBox.critical(self.ui, "Error", f"Failed to delete brand: {str(e)}")
        else:
            QMessageBox.warning(self.ui, "Warning", "Please select a brand to delete!")

    def reload_models(self):
        """Reload models and domain mapping"""
        try:
            load_config(reload_targetlist=True)
            # Reinitialize Phishpedia
            self.phishpedia_cls = PhishpediaWrapper()
            QMessageBox.information(self.ui, "Success", "Models reloaded successfully!")
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Failed to reload models: {str(e)}")

    def add_logo(self):
        # Get selected brand item
        selected_item = self.ui.tree_widget.currentItem()
        
        if selected_item and not selected_item.parent():  # Ensure it's a brand (top-level directory)
            brand_name = selected_item.text(0)
            
            # Open file dialog
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(
                self.ui, "Select Logo Image", "",
                "PNG Images (*.png)", options=options)
            
            if file_name:
                # Get target path
                target_dir = os.path.join('models/expand_targetlist', brand_name)
                base_name = os.path.basename(file_name)
                target_file = os.path.join(target_dir, base_name)

                # Check if file already exists
                if os.path.exists(target_file):
                    QMessageBox.warning(self.ui, "Warning", f"A logo with name '{base_name}' already exists!")
                    return
                
                try:
                    # Copy file to target directory
                    shutil.copy2(file_name, target_file)
                    # Update tree view
                    logo_item = QTreeWidgetItem([base_name])
                    selected_item.addChild(logo_item)
                except Exception as e:
                    QMessageBox.critical(self.ui, "Error", f"Failed to add logo: {str(e)}")
        else:
            QMessageBox.warning(self.ui, "Warning", "Please select a brand first!")

    def delete_logo(self):
        # Get selected item
        selected_item = self.ui.tree_widget.currentItem()
        
        if selected_item and selected_item.parent():  # Ensure it's a logo (child item)
            brand_name = selected_item.parent().text(0)
            logo_name = selected_item.text(0)
            
            # Confirm deletion
            reply = QMessageBox.question(
                self.ui,
                'Confirm Delete',
                f'Are you sure you want to delete logo "{logo_name}"?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Build file path
                logo_path = os.path.join('models/expand_targetlist', brand_name, logo_name)
                
                try:
                    # Delete file
                    os.remove(logo_path)
                    # Remove from tree view
                    selected_item.parent().removeChild(selected_item)
                except Exception as e:
                    QMessageBox.critical(self.ui, "Error", f"Failed to delete logo: {str(e)}")
        else:
            QMessageBox.warning(self.ui, "Warning", "Please select a logo to delete!")

    def domain_map_add(self, brand_name: str, domains_str: str) -> bool:
        """Add brand and domains to domain_map.pkl
        Args:
            brand_name: Brand name
            domains_str: Domain string, multiple domains separated by commas
        Returns:
            bool: Whether the addition was successful
        """
        try:
            domain_map_path = 'models/domain_map.pkl'
            
            # Process domain string, split and clean whitespace
            domains = [domain.strip() for domain in domains_str.split(',') if domain.strip()]
            
            if not domains:
                QMessageBox.warning(self.ui, "Warning", "Please enter at least one valid domain!")
                return False
            
            # Load existing domain mapping
            with open(domain_map_path, 'rb') as f:
                domain_map = pickle.load(f)
            
            # Add new brand and domains
            if brand_name in domain_map:
                if isinstance(domain_map[brand_name], list):
                    # Add new domains, avoid duplicates
                    existing_domains = set(domain_map[brand_name])
                    for domain in domains:
                        if domain not in existing_domains:
                            domain_map[brand_name].append(domain)
                else:
                    # If current value is not a list, convert to list
                    old_domain = domain_map[brand_name]
                    domain_map[brand_name] = [old_domain] + [d for d in domains if d != old_domain]
            else:
                domain_map[brand_name] = domains
            
            # Save updated mapping
            with open(domain_map_path, 'wb') as f:
                pickle.dump(domain_map, f)
            
            # Display added domains
            domains_added = '\n'.join(f"  - {d}" for d in domains)
            QMessageBox.information(
                self.ui,
                "Success",
                f"Added the following domains to brand '{brand_name}':\n{domains_added}"
            )
            
            return True
            
        except Exception as e:
            QMessageBox.critical(
                self.ui,
                "Error",
                f"Failed to update domain mapping: {str(e)}"
            )
            return False

    def domain_map_delete(self, brand_name: str) -> bool:
        """Delete brand and its domains from domain_map.pkl
        Args:
            brand_name: Brand name to delete
        Returns:
            bool: Whether the deletion was successful
        """
        try:
            domain_map_path = 'models/domain_map.pkl'
            
            # Load existing domain mapping
            with open(domain_map_path, 'rb') as f:
                domain_map = pickle.load(f)
            
            # Delete brand and its domains
            if brand_name in domain_map:
                del domain_map[brand_name]
            
            # Save updated mapping
            with open(domain_map_path, 'wb') as f:
                pickle.dump(domain_map, f)
            
            return True
            
        except Exception as e:
            QMessageBox.critical(
                self.ui,
                "Error",
                f"Failed to delete domain mapping: {str(e)}"
            )
            return False

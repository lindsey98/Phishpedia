// sidebar.js
new Vue({
    el: '#sidebar',
    data() {
        return {
            selectedDirectory: null, // currently selected directory
            selectedFile: null,      // currently selected file
            selectedDirectoryName: '',
            selectedFileName: '',
            showAddBrandForm: false, // toggle the add-brand form
            brandName: '',           // brand name
            brandDomain: '',         // brand domain
        }
    },
    mounted() {
        // Load the file tree when the page loads
        this.fetchFileTree();
        document.getElementById('logo-file-input').addEventListener('change', this.handleLogoFileSelect);

        const sidebar = document.getElementById("sidebar");
        const sidebarToggle = document.getElementById("sidebar-toggle");
        const closeSidebar = document.getElementById("close-sidebar");

        // Open the sidebar
        sidebarToggle.addEventListener("click", () => {
            sidebar.classList.add("open");
        });

        // Close the sidebar
        closeSidebar.addEventListener("click", () => {
            sidebar.classList.remove("open");
            this.clearSelected();
        });

        // Close the sidebar when clicking outside of it
        document.addEventListener("click", (event) => {
            if (!sidebar.contains(event.target) && !sidebarToggle.contains(event.target)) {
                sidebar.classList.remove("open");
                this.clearSelected();
            }
        });
    },
    methods: {
        // Recursively render the file tree
        renderFileTree(directory, parentPath = '') {
            // Get the file-tree container
            const fileTreeRoot = document.getElementById('file-tree-root');
            fileTreeRoot.innerHTML = ''; // clear existing content

            // Recursively build a file-tree node
            const createFileTreeNode = (item, parentPath) => {
                const li = document.createElement('li');
                li.classList.add('file-item');

                const currentPath = parentPath ? `${parentPath}/${item.name}` : item.name;

                if (item.type === 'directory') {
                    li.classList.add('file-folder');

                    const folderNameContainer = document.createElement('div');
                    folderNameContainer.classList.add('folder-name');
                    folderNameContainer.innerHTML = `<i class="folder-icon">📁</i><span>${item.name}</span>`;
                    li.appendChild(folderNameContainer);

                    if (item.children) {
                        const ul = document.createElement('ul');
                        ul.classList.add('hidden'); // hide subdirectories by default
                        item.children.forEach((child) => {
                            ul.appendChild(createFileTreeNode(child, currentPath)); // pass the current path
                        });
                        li.appendChild(ul);

                        // Single click selects the directory
                        folderNameContainer.addEventListener('click', (e) => {
                            e.stopPropagation();
                            this.selectDirectory(e, item.name);
                        });

                        // Double click expands/collapses the directory
                        folderNameContainer.addEventListener('dblclick', (e) => {
                            e.stopPropagation();
                            ul.classList.toggle('hidden');
                        });
                    }
                } else {
                    li.classList.add('file-file');
                    li.innerHTML = `<i class="file-icon">📄</i><span>${item.name}</span>`;

                    // Single click selects the file
                    li.addEventListener('click', (event) => {
                        this.selectFile(event, item.name, parentPath);
                    });
                }

                return li;
            };

            // Iterate over the top-level files and directories
            directory.forEach((item) => {
                fileTreeRoot.appendChild(createFileTreeNode(item, parentPath));
            });
        },
        // Fetch the file-tree data
        fetchFileTree() {
            // Request the file-tree data from the backend
            fetch('/get-directory') // backend file-tree endpoint
                .then((response) => response.json())
                .then((data) => {
                    if (data.file_tree) {
                        this.fileTree = data.file_tree; // store the file-tree data
                        this.renderFileTree(this.fileTree); // render the file tree
                    } else {
                        console.error('Invalid file tree data');
                        alert('Failed to load the file tree.');
                    }
                })
                .catch((error) => {
                    console.error('Error fetching file tree:', error);
                    alert('Unable to load the file tree, please try again later.');
                });
        },

        // Select a directory
        selectDirectory(event, directoryName) {
            const folderNameContainer = event.currentTarget;

            if (this.selectedDirectory) {
                this.selectedDirectory.classList.remove('selected');
            }
            if (this.selectedFile) {
                this.selectedFile.classList.remove('selected');
            }

            // Set the currently selected directory
            this.selectedDirectory = folderNameContainer;
            this.selectedDirectoryName = directoryName;
            folderNameContainer.classList.add('selected');
            this.selectedFile = null;
            this.selectedFileName = '';
        },

        // Select a file
        selectFile(event, fileName, parentPath) {
            const fileElement = event.currentTarget;

            if (this.selectedDirectory) {
                this.selectedDirectory.classList.remove('selected');
            }
            if (this.selectedFile) {
                this.selectedFile.classList.remove('selected');
            }

            // Set the currently selected file
            this.selectedFile = fileElement;
            this.selectedFileName = fileName;
            fileElement.classList.add('selected');
            this.selectedDirectory = null;
            this.selectedDirectoryName = parentPath;
        },

        // Add a brand
        addBrand() {
            this.showAddBrandForm = true;
        },

        // Close the add-brand form
        closeAddBrandForm() {
            this.showAddBrandForm = false;
            this.brandName = '';
            this.brandDomain = '';
        },

        // Submit the add-brand form
        submitAddBrandForm() {
            if (!this.brandName || !this.brandDomain) {
                alert('Please fill in all fields.');
                this.closeAddBrandForm();
                return;
            }

            const formData = new FormData();
            formData.append('brandName', this.brandName);
            formData.append('brandDomain', this.brandDomain);

            fetch('/add-brand', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Brand added successfully.');
                        this.fetchFileTree();
                        this.closeAddBrandForm();
                    } else {
                        alert('Failed to add brand: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Failed to add brand, please try again.');
                });
        },

        // Delete a brand
        delBrand() {
            if (this.selectedDirectory == null) {
                alert('Please select a brand first.');
                return;
            }
            const formData = new FormData();
            formData.append('directory', this.selectedDirectoryName);

            fetch('/del-brand', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    directory: this.selectedDirectoryName
                })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Brand deleted successfully.');
                        this.fetchFileTree();
                    }
                })
        },

        // Add a logo
        addLogo() {
            console.log('addLogo');
            if (this.selectedDirectory == null) {
                alert('Please select a brand first.');
                return;
            }
            document.getElementById('logo-file-input').click();
        },

        handleLogoFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                const formData = new FormData();
                formData.append('logo', file);
                formData.append('directory', this.selectedDirectoryName);

                fetch('/add-logo', {
                    method: 'POST',
                    body: formData
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            this.fetchFileTree();
                        } else {
                            alert('Failed to add logo: ' + data.error);
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Failed to add logo, please try again.');
                    });
            }
        },

        // Delete a logo
        delLogo() {
            if (this.selectedFile == null) {
                alert('Please select a logo first.');
                return;
            }

            const formData = new FormData();
            formData.append('directory', this.selectedDirectoryName);
            formData.append('filename', this.selectedFileName);

            fetch('/del-logo', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        this.fetchFileTree();
                    } else {
                        alert('Failed to delete logo: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Failed to delete logo, please try again.');
                });
        },

        async reloadModel() {
            const overlay = document.getElementById('overlay');

            overlay.style.display = 'flex';

            try {
                const response = await fetch('/reload-model', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                const data = await response.json();
            } catch (error) {
                alert('Failed to reload model.');
            } finally {
                overlay.style.display = 'none';
            }
        },

        clearSelected() {
            if (this.selectedDirectory) {
                this.selectedDirectory.classList.remove('selected');
                this.selectDirectory = null;
            }
            if (this.selectedFile) {
                this.selectedFile.classList.remove('selected');
                this.selectFile = null;
            }
            this.selectedDirectoryName = '';
            this.selectedFileName = '';
        },
    }
});
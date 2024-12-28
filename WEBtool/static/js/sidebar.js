// sidebar.js
new Vue({
    el: '#sidebar',
    data() {
        return {
            selectedDirectory: null, // 记录当前选中的目录
            selectedFile: null,      // 记录当前选中的文件
            selectedDirectoryName: '',
            selectedFileName: '',
            showAddBrandForm: false, // 控制表单显示与隐藏
            brandName: '',           // 品牌名称
            brandDomain: '',         // 品牌域名
        }
    },
    mounted() {
        // 网页加载时调用 fetchFileTree 函数
        this.fetchFileTree();
        document.getElementById('logo-file-input').addEventListener('change', this.handleLogoFileSelect);

        const sidebar = document.getElementById("sidebar");
        const sidebarToggle = document.getElementById("sidebar-toggle");
        const closeSidebar = document.getElementById("close-sidebar");

        // 点击打开侧边栏
        sidebarToggle.addEventListener("click", () => {
            sidebar.classList.add("open");
        });

        // 点击关闭侧边栏
        closeSidebar.addEventListener("click", () => {
            sidebar.classList.remove("open");
            this.clearSelected();
        });

        // 点击侧边栏外部关闭
        document.addEventListener("click", (event) => {
            if (!sidebar.contains(event.target) && !sidebarToggle.contains(event.target)) {
                sidebar.classList.remove("open");
                this.clearSelected();
            }
        });
    },
    methods: {
        // 递归渲染文件树
        renderFileTree(directory, parentPath = '') {
            // 获取文件树容器
            const fileTreeRoot = document.getElementById('file-tree-root');
            fileTreeRoot.innerHTML = ''; // 清空现有内容

            // 递归生成文件树节点
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
                        ul.classList.add('hidden'); // 默认隐藏子目录
                        item.children.forEach((child) => {
                            ul.appendChild(createFileTreeNode(child, currentPath)); // 传递当前目录的路径
                        });
                        li.appendChild(ul);

                        // 单击选中目录
                        folderNameContainer.addEventListener('click', (e) => {
                            e.stopPropagation();
                            this.selectDirectory(e, item.name);
                        });

                        // 双击展开/隐藏目录
                        folderNameContainer.addEventListener('dblclick', (e) => {
                            e.stopPropagation();
                            ul.classList.toggle('hidden');
                        });
                    }
                } else {
                    li.classList.add('file-file');
                    li.innerHTML = `<i class="file-icon">📄</i><span>${item.name}</span>`;

                    // 单击选中文件
                    li.addEventListener('click', (event) => {
                        this.selectFile(event, item.name, parentPath);
                    });
                }

                return li;
            };

            // 遍历顶层文件和目录
            directory.forEach((item) => {
                fileTreeRoot.appendChild(createFileTreeNode(item, parentPath));
            });
        },
        // 获取文件树数据
        fetchFileTree() {
            // 发送请求获取文件树数据
            fetch('/get-directory') // 后端文件树接口
                .then((response) => response.json())
                .then((data) => {
                    if (data.file_tree) {
                        this.fileTree = data.file_tree; // 存储文件树数据
                        this.renderFileTree(this.fileTree); // 渲染文件树
                    } else {
                        console.error('Invalid file tree data');
                        alert('文件树加载失败');
                    }
                })
                .catch((error) => {
                    console.error('Error fetching file tree:', error);
                    alert('无法加载文件树，请稍后重试。');
                });
        },

        // 选中目录
        selectDirectory(event, directoryName) {
            const folderNameContainer = event.currentTarget;

            if (this.selectedDirectory) {
                this.selectedDirectory.classList.remove('selected');
            }
            if (this.selectedFile) {
                this.selectedFile.classList.remove('selected');
            }

            // 设置当前选中的目录
            this.selectedDirectory = folderNameContainer;
            this.selectedDirectoryName = directoryName;
            folderNameContainer.classList.add('selected');
            this.selectedFile = null;
            this.selectedFileName = '';
        },

        // 选中文件
        selectFile(event, fileName, parentPath) {
            const fileElement = event.currentTarget;

            if (this.selectedDirectory) {
                this.selectedDirectory.classList.remove('selected');
            }
            if (this.selectedFile) {
                this.selectedFile.classList.remove('selected');
            }

            // 设置当前选中的文件
            this.selectedFile = fileElement;
            this.selectedFileName = fileName;
            fileElement.classList.add('selected');
            this.selectedDirectory = null;
            this.selectedDirectoryName = parentPath;
        },

        // 增加品牌
        addBrand() {
            this.showAddBrandForm = true;
        },

        // 关闭添加品牌的表单
        closeAddBrandForm() {
            this.showAddBrandForm = false;
            this.brandName = '';
            this.brandDomain = '';
        },

        // 提交添加品牌的表单
        submitAddBrandForm() {
            if (!this.brandName || !this.brandDomain) {
                alert('Please fill in all fields.');
                closeAddBrandForm()
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

        // 删除品牌
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
                        alert('Brand deletedsuccessfully.');
                        this.fetchFileTree();
                    }
                })
        },

        // 增加logo
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

        // 删除logo
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
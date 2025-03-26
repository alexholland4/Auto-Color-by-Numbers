document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('imageInput');
    const uploadMsg = document.getElementById('uploadMsg');
    const uploadedImage = document.getElementById('uploadedImage');
    const startOverBtn = document.getElementById('startOverBtn');
    const imageAndOptions = document.querySelector('.image-and-options');
    const processBtn = document.getElementById('processBtn');
    const logDiv = document.getElementById('log');
    const logSection = document.querySelector('.log-section');
    const processedSection = document.querySelector('.processed-section');
    const masterListSection = document.querySelector('.master-list-section');
    
    let uploadedFilename = '';
  
    // 1) Automatically upload on file selection
    imageInput.addEventListener('change', function() {
      const file = imageInput.files[0];
      if (!file) return;
      const formData = new FormData();
      formData.append('image', file);
      
      fetch('/upload', {
        method: 'POST',
        body: formData
      })
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          uploadMsg.textContent = data.error;
        } else {
          uploadMsg.textContent = data.message;
          uploadedFilename = data.filename;
          // Show the uploaded image at fixed size
          uploadedImage.src = `/static/uploads/${uploadedFilename}`;
          uploadedImage.style.display = 'block';
          // Hide the file input so user can't upload more
          imageInput.style.display = 'none';
          // Show "Start Over" button
          startOverBtn.style.display = 'inline-block';
          // Show the main container (image + side panel)
          imageAndOptions.style.display = 'flex';
        }
      })
      .catch(err => console.error(err));
    });
  
    // 2) Start Over: reload the page
    startOverBtn.addEventListener('click', function() {
      window.location.reload();
    });
  
    // 3) Process image with SSE
    processBtn.addEventListener('click', function() {
      if (!uploadedFilename) {
        alert('Please upload an image first.');
        return;
      }
      // Get selected detail option
      const detailOption = document.querySelector('input[name="detail"]:checked').value;
      
      // Hide options, show log
      document.querySelector('.options-section').style.display = 'none';
      logSection.style.display = 'block';
      processedSection.style.display = 'none';
      masterListSection.style.display = 'none';
      logDiv.innerHTML = '';
  
      // Use SSE
      const evtSource = new EventSource('/process?' + new URLSearchParams({
        filename: uploadedFilename,
        detail_option: detailOption
      }));
      
      evtSource.onmessage = function(e) {
        try {
          // If data is JSON, we've finished processing
          const result = JSON.parse(e.data);
          // Hide log, show processed images side by side
          logSection.style.display = 'none';
          processedSection.style.display = 'flex';
          masterListSection.style.display = 'block';
          
          document.getElementById('numberedImg').src = '/' + result.numbered;
          document.getElementById('downloadNumbered').href = '/' + result.numbered;
          document.getElementById('coloredImg').src = '/' + result.colored;
          document.getElementById('downloadColored').href = '/' + result.colored;
          
          // Build master list with color squares
          const masterListDiv = document.getElementById('masterList');
          masterListDiv.innerHTML = '';
          for (const [label, rgb] of Object.entries(result.master_list)) {
            const item = document.createElement('div');
            item.className = 'item';
            const colorSquare = document.createElement('div');
            colorSquare.className = 'color-square';
            colorSquare.style.backgroundColor = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
            const text = document.createElement('span');
            text.textContent = ` Region ${label}: rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
            item.appendChild(colorSquare);
            item.appendChild(text);
            masterListDiv.appendChild(item);
          }
          evtSource.close();
        } catch (err) {
          // Otherwise, it's a log message
          logDiv.innerHTML += `<p>${e.data}</p>`;
          logDiv.scrollTop = logDiv.scrollHeight;
        }
      };
  
      evtSource.onerror = function() {
        logDiv.innerHTML += `<p>Error during processing.</p>`;
        evtSource.close();
      };
    });
  });
  
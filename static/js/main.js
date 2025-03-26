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
  
    // Automatically upload image when file selected
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
          // Display the uploaded image (displayed in fixed dimensions via CSS)
          uploadedImage.src = `/static/uploads/${uploadedFilename}`;
          uploadedImage.style.display = 'block';
          // Hide the file input so upload options disappear
          imageInput.style.display = 'none';
          // Show the Start Over button
          startOverBtn.style.display = 'inline-block';
          // Show the main content container (original image + side panel)
          imageAndOptions.style.display = 'flex';
        }
      })
      .catch(err => console.error(err));
    });
  
    // Start Over reloads the page
    startOverBtn.addEventListener('click', function() {
      window.location.reload();
    });
  
    processBtn.addEventListener('click', function() {
      if (!uploadedFilename) {
        alert('Please upload an image first.');
        return;
      }
      // Get selected detail option
      const detailOption = document.querySelector('input[name="detail"]:checked').value;
      
      // Hide the options panel and show processing log
      document.querySelector('.options-section').style.display = 'none';
      logSection.style.display = 'block';
      processedSection.style.display = 'none';
      masterListSection.style.display = 'none';
      logDiv.innerHTML = '';
  
      // Start processing via SSE
      const evtSource = new EventSource('/process?' + new URLSearchParams({
        filename: uploadedFilename,
        detail_option: detailOption
      }));
      
      evtSource.onmessage = function(e) {
        try {
          const result = JSON.parse(e.data);
          // Processing finished, hide log and show processed images
          logSection.style.display = 'none';
          processedSection.style.display = 'flex';
          masterListSection.style.display = 'block';
          
          document.getElementById('numberedImg').src = '/' + result.numbered;
          document.getElementById('downloadNumbered').href = '/' + result.numbered;
          document.getElementById('coloredImg').src = '/' + result.colored;
          document.getElementById('downloadColored').href = '/' + result.colored;
          
          // Build the master list display
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
          // Otherwise, treat the message as a log update
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
  
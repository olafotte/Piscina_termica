
// Garantir que initThreeJS está definido globalmente
//if (typeof window.initThreeJS !== 'function') {
//    window.initThreeJS = function() {
//        // Exemplo de inicialização segura
//        const container = document.getElementById('container');
//        if (!container) {
//            console.error("Elemento com id 'container' não encontrado!");
//            return;
//        }
//        // Aqui você pode colocar o código real do Three.js
//        console.warn('initThreeJS stub chamada. Implemente a função para inicializar o Three.js.');
//    };
//}

// Garante que a inicialização só ocorre após o DOM estar pronto
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        if (typeof window.initThreeJS === 'function') {
            window.initThreeJS();
        }
    });
} else {
    if (typeof window.initThreeJS === 'function') {
        window.initThreeJS();
    }
}

function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; }
            tablinks = document.getElementsByClassName("tab-button");
            for (i = 0; i < tablinks.length; i++) { tablinks[i].className = tablinks[i].className.replace(" active", ""); }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        
        
        document.querySelector('form').addEventListener('submit', function() {
            document.getElementById('loading-overlay').style.display = 'flex';
        });

        function copyReport() {
            const textarea = document.getElementById('report-text');
            navigator.clipboard.writeText(textarea.value).then(() => {
                alert('Relatório copiado para a área de transferência!');
            }).catch(err => {
                console.error('Falha ao copiar o relatório: ', err);
                alert('Não foi possível copiar o relatório.');
            });
        }



            let config; // Declare a variável config para ser usada globalmente
    let threeJSInitialized = false; // New global flag

        function showDailyForMonth(yearMonth) {
            const [year, month] = yearMonth.split('-');
            const filteredDailyData = allDailyData.filter(row => {
                const rowDate = new Date(row.timestamp);
                return rowDate.getFullYear() == parseInt(year) && (rowDate.getMonth() + 1) == parseInt(month);
            });
            populateDailyTable(filteredDailyData);

            // Activate the 'diario' tab
            const diarioTabButton = document.getElementById('diario-tab-button');
            if (diarioTabButton) {
                diarioTabButton.click();
            }
        }

        function showHourlyForDay(yearMonthDay) {
            console.log("showHourlyForDay called with yearMonthDay:", yearMonthDay);
            console.log("allHourlyData length:", allHourlyData.length);

            // Garante que yearMonthDay está no formato YYYY-MM-DD
            let normalizedYearMonthDay = yearMonthDay;
            if (yearMonthDay.length > 10) {
                // Tenta extrair a data se vier em formato estranho
                const match = yearMonthDay.match(/\d{4}-\d{2}-\d{2}/);
                if (match) normalizedYearMonthDay = match[0];
            }

            const filteredHourlyData = allHourlyData.filter(row => {
                const dateObj = new Date(row.timestamp);
                const year = dateObj.getFullYear();
                const month = (dateObj.getMonth() + 1).toString().padStart(2, '0');
                const day = dateObj.getDate().toString().padStart(2, '0');
                const rowDate = `${year}-${month}-${day}`;
                // Comparação apenas pelo dia
                //console.log(`Comparing: row.timestamp=${row.timestamp} (parsed to ${rowDate}) with yearMonthDay=${normalizedYearMonthDay}`);
                return rowDate === normalizedYearMonthDay;
            });
            console.log("Filtered hourly data length:", filteredHourlyData.length);
            populateHourlyTable(filteredHourlyData);

            // Activate the 'horario' tab
            const horarioTabButton = document.querySelector('.tab-button[data-tab-name="horario"]');
            if (horarioTabButton) {
                horarioTabButton.click();
            }
        }

        function populateHourlyTable(data) {
            console.log("populateHourlyTable called with data length:", data.length);
            const tableBody = document.querySelector('#hourly-table tbody');
            if (!tableBody) {
                console.error("Error: #hourly-table tbody not found!");
                return;
            }
            let tableHtml = '';
            data.forEach(row => {
                const timestamp = new Date(row.timestamp);
                tableHtml += `
                    <tr style="cursor: pointer;" onclick="
                        const datetimeInput = document.getElementById('datetime-input');
                        console.log('Clicked hourly row. Timestamp:', '${row.timestamp}');
                        const date = new Date('${row.timestamp}');
                        const year = date.getFullYear();
                        const month = (date.getMonth() + 1).toString().padStart(2, '0');
                        const day = date.getDate().toString().padStart(2, '0');
                        const hours = date.getHours().toString().padStart(2, '0');
                        const minutes = date.getMinutes().toString().padStart(2, '0');
                        const formattedDatetime = year + '-' + month + '-' + day + ' ' + hours + ':' + minutes;
                        console.log('Value to set to datetime-input:', formattedDatetime);
                        if (datetimeInput) {
                            datetimeInput.value = formattedDatetime;
                            console.log('Dispatching change event for datetime-input.');
                            datetimeInput.dispatchEvent(new Event('change'));
                            console.log('datetime-input value set and change event dispatched.');
                        } else {
                            console.error('datetime-input not found!');
                        }
                    ">
                        <td>${timestamp.toLocaleDateString('pt-BR', {timeZone: 'America/Sao_Paulo'})} ${timestamp.toLocaleTimeString('pt-BR', {hour: '2-digit', minute:'2-digit', timeZone: 'America/Sao_Paulo'})}</td>
                        <td>R$ ${row.cost_hour !== undefined ? row.cost_hour.toFixed(2).replace('.', ',') : '-'}</td>
                        <td>${row.consumed_kwh !== undefined ? row.consumed_kwh.toFixed(3).replace('.', ',') : '-'}</td>
                        <td>${row.hourly_shading !== undefined ? row.hourly_shading.toFixed(2).replace('.', ',') : '-'}</td>
                        <td>${row.heater_gain_Wh !== undefined ? row.heater_gain_Wh.toFixed(1).replace('.', ',') : '-'}</td>
                        <td>${row.solar_gain_Wh !== undefined ? row.solar_gain_Wh.toFixed(1).replace('.', ',') : '-'}</td>
                        <td>${row.evaporation_loss_Wh !== undefined ? row.evaporation_loss_Wh.toFixed(1).replace('.', ',') : '-'}</td>
                        <td>${row.water_temp_start !== undefined ? row.water_temp_start.toFixed(2).replace('.', ',') : '-'}</td>
                        <td>${row.water_temp_end !== undefined ? row.water_temp_end.toFixed(2).replace('.', ',') : '-'}</td>
                    </tr>
                `;
            });
            tableBody.innerHTML = tableHtml;
        }

        function populateDailyTable(data) {
            const tableBody = document.querySelector('#diario tbody');
            tableBody.innerHTML = ''; // Clear existing rows
            data.forEach(row => {
                const tr = document.createElement('tr');
                const timestamp = new Date(row.timestamp);
                tr.innerHTML = `
                    <td>${timestamp.toLocaleDateString('pt-BR', {timeZone: 'America/Sao_Paulo'})}</td>
                    <td>R$ ${row.cost_hour.toFixed(2).replace('.', ',')}</td>
                    <td>${row.consumed_kwh.toFixed(0).replace('.', ',')}</td>
                    <td>${row.hourly_shading.toFixed(2).replace('.', ',')}</td>
                `;
                // --- INÍCIO DA ALTERAÇÃO ---
                tr.style.cursor = 'pointer'; // Adiciona um feedback visual de que a linha é clicável
                tr.addEventListener('click', () => {
                    // Chama a função existente para filtrar e mostrar os dados horários
                    const dateObj = new Date(row.timestamp);
                    const year = dateObj.getFullYear();
                    const month = (dateObj.getMonth() + 1).toString().padStart(2, '0');
                    const day = dateObj.getDate().toString().padStart(2, '0');
                    const dateStr = `${year}-${month}-${day}`;
                    showHourlyForDay(dateStr); // Passa a data no formato YYYY-MM-DD
                });
                // --- FIM DA ALTERAÇÃO ---
                tableBody.appendChild(tr);
            });
        }

        // Modifique o evento DOMContentLoaded
        document.addEventListener('DOMContentLoaded', () => {
            // Add event listeners to tab buttons
            document.querySelectorAll('.tab-button').forEach(button => {
                button.addEventListener('click', (evt) => {
                    const tabName = evt.currentTarget.getAttribute('data-tab-name');
                    openTab(evt, tabName);
                    // If the 'Hora a Hora' tab is opened, populate it with all hourly data
                    if (tabName === 'horario') {
                        // populateHourlyTable(allHourlyData); // Removed this line
                    }
                });
            });

            // Programmatically click the first tab to open it
            if (document.getElementsByClassName("tab-button").length > 0) {
                document.getElementsByClassName("tab-button")[0].click();
            }

            // Pega o elemento que contém os dados
            const dataContainer = document.getElementById('data-container');
            // Verifica se o elemento e o atributo de dados existem
            if (dataContainer && dataContainer.dataset.hourlyData) {
                try {
                    // Converte a string JSON do atributo para um objeto e popula a variável
                    allHourlyData = JSON.parse(dataContainer.dataset.hourlyData);
                } catch (e) {
                    console.error("Erro ao processar dados horários:", e);
                }
            }
            if (dataContainer && dataContainer.dataset.dailyData) {
                try {
                    allDailyData = JSON.parse(dataContainer.dataset.dailyData);
                } catch (e) {
                    console.error("Erro ao processar dados diários:", e);
                }
            }
        });

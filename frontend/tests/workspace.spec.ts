import {test,expect} from '@playwright/test';

test('map panels, time context, navigation, scenarios and evidence',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');
 await expect(page.getByTestId('world-map')).toBeVisible();
 await expect(page.locator('.insight-hero')).toBeVisible();
 await page.screenshot({path:'test-results/hydra-world.png',fullPage:true});
 await page.getByRole('button',{name:'Minimize Layers',exact:true}).click();
 await page.getByRole('button',{name:'Minimize AI Insights',exact:true}).click();
 await expect(page.locator('.layer-panel')).toHaveCount(0);
 await expect(page.locator('.insight-panel')).toHaveCount(0);
 await page.getByRole('button',{name:'Expand Layers',exact:true}).click();
 await page.getByRole('button',{name:'Expand AI Insights',exact:true}).click();
 await page.getByRole('button',{name:'+48h DAY 2'}).click();
 await expect(page.locator('.context-location')).toContainText('2025-12-30');
 await page.getByLabel('Search layers').fill('disagreement');
 await page.getByRole('button',{name:'Model disagreement',exact:true}).click();
 await expect(page.getByRole('button',{name:'✓ Model disagreement',exact:true})).toHaveAttribute('aria-pressed','true');
 await page.getByLabel('Search layers').fill('');
 for(const name of ['Situation','Models','Extreme Events','Impact & Risk','Verification','Sectors','Scenario','WeatherGPT','Operations']){
  await page.locator('.taskbar').getByRole('button',{name,exact:true}).click();
  await expect(page.getByRole('dialog',{name,exact:true})).toBeVisible();
  await page.getByLabel('Close dialog').click();
 }
 await page.locator('.taskbar').getByRole('button',{name:'Scenario',exact:true}).click();
 await page.getByRole('button',{name:'Run scenario',exact:true}).click();
 await expect(page.getByRole('heading',{name:/SIMULATED SCENARIO/})).toBeVisible();
 await page.getByLabel('Close dialog').click();
 await page.locator('.taskbar').getByRole('button',{name:'WeatherGPT',exact:true}).click();
 await page.getByRole('button',{name:'Explain forecast uncertainty',exact:true}).click();
 await expect(page.locator('.chat-answer').last()).toContainText(/interval|uncertainty/i);
 await page.getByLabel('Close dialog').click();
 await page.keyboard.press('Control+k');
 await page.getByLabel('Command search').fill('Kochi');
 await page.getByRole('button',{name:'Kochi, Kerala',exact:true}).click();
 await expect(page.getByRole('dialog',{name:'Location',exact:true})).toContainText('No supplied forecast');
 await page.getByLabel('Close dialog').click();
 await page.getByLabel('Forecast source').selectOption('ecmwf');
 await expect(page.locator('.insight-hero')).toBeVisible();
 await expect(page.locator('.context-location')).toContainText('2026-09-24');
 await page.screenshot({path:'test-results/hydra-desktop.png',fullPage:true});
 expect(errors).toEqual([]);
});

test('tablet keeps map and dialogs usable',async({page})=>{
 await page.setViewportSize({width:768,height:1024});await page.goto('/');
 await page.getByLabel('Minimize Layers',{exact:true}).click();await page.getByLabel('Minimize AI Insights',{exact:true}).click();
 await expect(page.getByTestId('world-map')).toBeVisible();
 await page.locator('.taskbar').getByRole('button',{name:'Models',exact:true}).click();
 await expect(page.getByRole('dialog',{name:'Models',exact:true})).toBeVisible();
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
 await page.screenshot({path:'test-results/hydra-tablet.png',fullPage:true});
});

test('India states and distinct layer notations work together',async({page})=>{
 test.setTimeout(90000);
 await page.goto('/');
 await expect(page.getByLabel('Select India state').locator('option')).toHaveCount(37,{timeout:30000});
 await page.getByRole('button',{name:'INDIA',exact:true}).click();
 await page.getByLabel('Forecast source').selectOption('ecmwf');
 await expect(page.locator('.map-caption')).toContainText('Select an India state');
 const stateField=page.waitForResponse(response=>response.url().includes('/api/field?source=ecmwf')&&response.url().includes('state=Kerala'));
 await page.getByLabel('Select India state').selectOption('Kerala');
 expect((await stateField).ok()).toBeTruthy();
 await expect(page.getByLabel('Select India state')).toHaveValue('Kerala');
 await expect(page.locator('.state-insight-panel .context-location')).toContainText('Kerala',{timeout:30000});
 await expect(page.locator('.map-caption')).not.toContainText('Select an India state');
 await page.getByLabel('Search layers').fill('Current city weather');
 await page.getByRole('button',{name:'Current city weather',exact:true}).click();
 await expect(page.getByRole('dialog',{name:'Observations',exact:true})).toContainText('Current weather');
 await page.getByLabel('Close dialog').click();
 for(const name of ['Grid-value heatmap','Temperature grid','Wind notation']){
  await page.getByLabel('Search layers').fill(name);
  await page.getByRole('button',{name,exact:true}).click();
 }
 await page.getByLabel('Search layers').fill('');
 await expect(page.locator('.layer-legend')).toContainText('Grid heatmap');
 await expect(page.locator('.layer-legend')).toContainText('Temperature cells');
 await expect(page.locator('.layer-legend')).toContainText('Wind arrows');
 const map=await page.getByTestId('world-map').boundingBox();
 if(!map)throw new Error('Map bounds unavailable');
 await page.mouse.move(map.x+map.width/2,map.y+map.height/2);
 await expect(page.locator('.grid-inspector')).toContainText('NEAREST DATA GRID');
 await page.getByLabel('Search layers').fill('Live weather');
 await page.getByRole('button',{name:'Live weather locations',exact:true}).click();
 await expect(page.locator('.layer-legend')).toContainText('Live weather locations');
 await expect(page.getByRole('status')).toContainText('Open-Meteo model data');
 for(const name of ['Open-Meteo current rainfall','Open-Meteo 24h rain outlook','Open-Meteo humidity','Thunderstorm potential','Wind-gust risk','Heat stress','Soil moisture / crop stress']){
  await page.getByLabel('Search layers').fill(name);
  await page.getByRole('button',{name,exact:true}).click();
 }
 await expect(page.locator('.layer-legend')).toContainText('Open-Meteo current rainfall');
 await expect(page.locator('.layer-legend')).toContainText('Open-Meteo 24h rain outlook');
 await expect(page.locator('.layer-legend')).toContainText('Open-Meteo humidity');
 await expect(page.locator('.layer-legend')).toContainText('Thunderstorm potential');
 await expect(page.locator('.layer-legend')).toContainText('Wind-gust risk');
 await expect(page.locator('.layer-legend')).toContainText('Heat stress');
 await expect(page.locator('.layer-legend')).toContainText('Soil moisture / crop stress');
 await page.mouse.move(map.x+map.width/2,map.y+map.height/2);
 await expect(page.locator('.grid-inspector')).toContainText('Open-Meteo current rainfall',{timeout:30000});
 await expect(page.locator('.grid-inspector')).toContainText('Open-Meteo next 24h rainfall');
 await expect(page.locator('.grid-inspector')).toContainText('Open-Meteo humidity');
 await expect(page.locator('.grid-inspector')).toContainText('CAPE');
 await expect(page.locator('.grid-inspector')).toContainText('Peak wind gust');
 await expect(page.locator('.grid-inspector')).toContainText('Peak wet-bulb temperature');
 await expect(page.locator('.grid-inspector')).toContainText('Soil moisture');
 await page.screenshot({path:'test-results/hydra-india-layers.png',fullPage:true});
});

test('radar and satellite controls request their public imagery',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');
 await expect(page.locator('canvas.maplibregl-canvas')).toBeVisible();
 await page.getByRole('button',{name:'3D',exact:true}).click();
 await expect(page.getByRole('button',{name:'3D',exact:true})).toHaveClass(/selected/);
 const radar=page.waitForResponse(response=>response.url().includes('/api/radar-frames'));
 await page.getByRole('button',{name:'RADAR',exact:true}).click();
 expect((await radar).ok()).toBeTruthy();
 await expect(page.getByRole('button',{name:'RADAR',exact:true})).toHaveClass(/selected/);
 await expect(page.locator('.layer-panel')).toBeVisible();
 await expect(page.getByRole('button',{name:'Radar precipitation',exact:true})).toBeVisible();
 const satellite=page.waitForResponse(response=>response.url().includes('/api/imd-satellite'));
 await page.getByRole('button',{name:'SATELLITE',exact:true}).click();
 expect((await satellite).ok()).toBeTruthy();
 await expect(page.getByRole('button',{name:'SATELLITE',exact:true})).toHaveClass(/selected/);
 await expect(page.locator('canvas.maplibregl-canvas')).toBeVisible();
 expect(errors).toEqual([]);
});
